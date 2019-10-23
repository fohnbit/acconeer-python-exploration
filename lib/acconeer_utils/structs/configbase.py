from weakref import WeakKeyDictionary
import json
from enum import Enum
import numpy as np
import os
from copy import copy

try:
    from PyQt5 import QtCore
    from PyQt5.QtWidgets import (
        QWidget,
        QFrame,
        QGridLayout,
        QDoubleSpinBox,
        QSlider,
        QLabel,
        QHBoxLayout,
        QVBoxLayout,
        QSpinBox,
        QCheckBox,
        QPushButton,
        QFileDialog,
        QComboBox,
    )
except ImportError:
    pass


class Pidget(QFrame):
    def __init__(self, param, parent_instance):
        super().__init__()

        self.param = param
        self._parent_instance = parent_instance

        self._enable_event_handler = False

        self.hide()

    def update(self):
        self._enable_event_handler = False
        self._update()
        self._enable_event_handler = True

    def _update(self):
        pass

    def _get_param_value(self):
        return self.param.__get__(self._parent_instance)

    def _subwidget_event_handler(self, val):
        if self._enable_event_handler:
            self.param.pidget_event_handler(self._parent_instance, val)


class PidgetStub(Pidget):
    def __init__(self, param, parent_instance):
        super().__init__(param, parent_instance)

        self.setObjectName("frame")
        self.default_css = "#frame {border: 1px solid lightgrey; border-radius: 3px;}"
        self.setStyleSheet(self.default_css)

        if param.help:
            self.setToolTip(param.help)

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 6, 0, 6)

        self.grid_widget = QWidget()
        self.layout.addWidget(self.grid_widget)

        self.grid = QGridLayout(self.grid_widget)
        self.grid.setContentsMargins(6, 0, 6, 0)

        self.error_frame = QFrame()
        self.error_frame.setStyleSheet(".QFrame {border-top: 1px solid lightgrey;}")
        error_layout = QHBoxLayout(self.error_frame)
        error_layout.setContentsMargins(6, 6, 6, 0)
        self.error_label = QLabel()
        self.error_label.setStyleSheet("color: orangered; font-weight: bold;")
        error_layout.addWidget(self.error_label)
        self.layout.addWidget(self.error_frame)
        self._unset_error()

    def _update(self):
        state = self._parent_instance._state

        if state != Config.State.UNLOADED:
            if callable(self.param.visible):
                visible = self.param.visible(self._parent_instance)
            else:
                visible = self.param.visible
        else:
            visible = False

        self.setVisible(bool(visible))

        self.setEnabled(state != Config.State.LIVE or self.param.is_live_updateable)

        try:
            self.param.recheck(self._parent_instance)
        except AttributeError:
            pass
        except ValueError as e:
            self._set_error(e)
        else:
            self._unset_error()

    def _set_error(self, e):
        self.error_label.setText(str(e))

        self.setStyleSheet((
                "#frame {"
                "background-color: bisque;"
                "border: 1px solid lightgrey;"
                "border-radius: 3px;"
                "}"
                ))

        self.error_frame.show()

    def _unset_error(self):
        self.error_frame.hide()
        self.setStyleSheet(self.default_css)

    def _subwidget_event_handler(self, val):
        try:
            super()._subwidget_event_handler(val)
        except ValueError as e:
            self._set_error(e)


class ComboBoxPidget(PidgetStub):
    def __init__(self, param, parent_instance):
        assert isinstance(param, EnumParameter)

        super().__init__(param, parent_instance)

        label = QLabel(param.label, self)
        self.grid.addWidget(label, 0, 0, 1, 1)

        self.cb = QComboBox()
        members = param.enum.__members__.values()
        label_attrib_name = "label" if hasattr(param.enum, "label") else "value"
        self.cb.addItems([getattr(e, label_attrib_name) for e in members])
        self.cb.currentIndexChanged.connect(self.__cb_event_handler)
        self.grid.addWidget(self.cb, 0, 1, 1, 1)

        self.update()

    def _update(self):
        super()._update()
        value = self._get_param_value()
        index = list(self.param.enum.__members__.values()).index(value)
        self.cb.setCurrentIndex(index)

    def __cb_event_handler(self, index):
        value = list(self.param.enum.__members__.values())[index]
        self._subwidget_event_handler(value)


class BoolCheckboxPidget(PidgetStub):
    def __init__(self, param, parent_instance):
        super().__init__(param, parent_instance)

        self.checkbox = QCheckBox(param.label, self)
        self.checkbox.setTristate(False)
        self.checkbox.stateChanged.connect(self._subwidget_event_handler)
        self.grid.addWidget(self.checkbox, 0, 0, 1, 1)

        self.update()

    def _update(self):
        super()._update()
        value = self._get_param_value()
        self.checkbox.setChecked(value)


class IntSpinBoxPidget(PidgetStub):
    def __init__(self, param, parent_instance):
        super().__init__(param, parent_instance)

        self.grid.setColumnStretch(0, 7)
        self.grid.setColumnStretch(1, 3)

        label = QLabel(self)
        suffix = " [{}]".format(param.unit) if param.unit else ""
        label.setText(param.label + suffix)
        self.grid.addWidget(label, 0, 0, 1, 1)

        self.spin_box = QSpinBox(self)
        self.spin_box.setSingleStep(param.step)
        self.spin_box.valueChanged.connect(self._subwidget_event_handler)
        self.spin_box.setKeyboardTracking(False)

        if param.limits is None:
            self.spin_box.setRange(-1e9, 1e9)
        else:
            self.spin_box.setRange(*param.limits)

        self.grid.addWidget(self.spin_box, 0, 1, 1, 1)

        self.update()

    def _update(self):
        super()._update()
        value = self._get_param_value()
        self.spin_box.setValue(value)


class FloatRangeSpinBoxesPidget(PidgetStub):
    def __init__(self, param, parent_instance):
        assert isinstance(param, FloatRangeParameter)

        super().__init__(param, parent_instance)

        self.grid.setColumnStretch(0, 1)
        self.grid.setColumnStretch(1, 1)

        label = QLabel(self)
        suffix = " [{}]".format(param.unit) if param.unit else ""
        label.setText(param.label + suffix)
        self.grid.addWidget(label, 0, 0, 1, 1)

        self.spin_boxes = []
        for i in range(2):
            spin_box = QDoubleSpinBox(self)
            spin_box.setDecimals(param.decimals)
            spin_box.setSingleStep(10**(-param.decimals))
            spin_box.valueChanged.connect(lambda v, i=i: self.__spin_box_event_handler(v, i))
            spin_box.setKeyboardTracking(False)
            spin_box.setMinimum(param.limits[0])
            spin_box.setMaximum(param.limits[1])
            self.grid.addWidget(spin_box, 1, i, 1, 1)
            self.spin_boxes.append(spin_box)

        self.update()

    def _update(self):
        super()._update()
        values = self._get_param_value()

        for spin_box, val in zip(self.spin_boxes, values):
            spin_box.setValue(val)

    def __spin_box_event_handler(self, val, index):
        vals = self._get_param_value()
        vals[index] = val
        self._subwidget_event_handler(vals)


class FloatSpinBoxPidget(PidgetStub):
    def __init__(self, param, parent_instance):
        assert isinstance(param, FloatParameter)

        super().__init__(param, parent_instance)

        self.grid.setColumnStretch(0, 7)
        self.grid.setColumnStretch(1, 3)

        label = QLabel(self)
        suffix = " [{}]".format(param.unit) if param.unit else ""
        label.setText(param.label + suffix)
        self.grid.addWidget(label, 0, 0, 1, 1)

        self.spin_box = QDoubleSpinBox(self)
        self.spin_box.setDecimals(param.decimals)
        self.spin_box.setSingleStep(10**(-param.decimals))
        self.spin_box.valueChanged.connect(self._subwidget_event_handler)
        self.spin_box.setKeyboardTracking(False)
        self.spin_box.setMinimum(param.limits[0])
        self.spin_box.setMaximum(param.limits[1])
        self.grid.addWidget(self.spin_box, 0, 1, 1, 1)

        self.update()

    def _update(self):
        super()._update()
        value = self._get_param_value()
        self.spin_box.setValue(value)


class FloatSpinBoxAndSliderPidget(PidgetStub):
    NUM_SLIDER_STEPS = 200

    def __init__(self, param, parent_instance):
        assert isinstance(param, FloatParameter)

        super().__init__(param, parent_instance)

        self.grid.setColumnStretch(0, 7)
        self.grid.setColumnStretch(1, 3)

        label = QLabel(self)
        suffix = " [{}]".format(param.unit) if param.unit else ""
        label.setText(param.label + suffix)
        self.grid.addWidget(label, 0, 0, 1, 1)

        self.spin_box = QDoubleSpinBox(self)
        self.spin_box.setDecimals(param.decimals)
        self.spin_box.setSingleStep(10**(-param.decimals))
        self.spin_box.valueChanged.connect(self._subwidget_event_handler)
        self.spin_box.setKeyboardTracking(False)
        self.spin_box.setMinimum(param.limits[0])
        self.spin_box.setMaximum(param.limits[1])
        self.grid.addWidget(self.spin_box, 0, 1, 1, 1)

        slider_widget = QWidget()
        slider_layout = QHBoxLayout(slider_widget)
        slider_layout.setContentsMargins(0, 0, 0, 0)
        slider_layout.addWidget(QLabel(str(param.limits[0])))
        self.slider = QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, self.NUM_SLIDER_STEPS)
        self.slider.sliderPressed.connect(self.__slider_event_handler)
        self.slider.valueChanged.connect(self.__slider_event_handler)
        slider_layout.addWidget(self.slider, 1)
        slider_layout.addWidget(QLabel(str(param.limits[1])))
        self.grid.addWidget(slider_widget, 1, 0, 1, 2)

        self.update()

    def _update(self):
        super()._update()
        value = self._get_param_value()
        self.spin_box.setValue(value)
        self.slider.setValue(self.__to_slider_scale(value))

    def __slider_event_handler(self, x=None):
        if x is None:
            x = self.slider.sliderPosition()

        self._subwidget_event_handler(self.__from_slider_scale(x))

    def __to_slider_scale(self, x):
        lower, upper = self.param.limits

        if self.param.logscale:
            lower = np.log(lower)
            upper = np.log(upper)
            x = np.log(x)

        y = (x - lower) / (upper - lower) * self.NUM_SLIDER_STEPS
        return int(round(y))

    def __from_slider_scale(self, y):
        lower, upper = self.param.limits

        if self.param.logscale:
            lower = np.log(lower)
            upper = np.log(upper)

        x = y / self.NUM_SLIDER_STEPS * (upper - lower) + lower

        if self.param.logscale:
            x = np.exp(x)

        return x


class Parameter:
    def __init__(self, **kwargs):
        self.label = kwargs.pop("label")
        self.is_live_updateable = kwargs.pop("updateable", False)
        self.does_dump = kwargs.pop("does_dump", False)
        self.pidget_class = kwargs.pop("pidget_class", None)
        self.pidget_location = kwargs.pop("pidget_location", None)
        self.order = kwargs.pop("order", -1)
        self.help = kwargs.pop("help", None)
        self.visible = kwargs.pop("visible", True)

        # don't care if unused
        kwargs.pop("default_value", None)

        if kwargs:
            a_key = next(iter(kwargs.keys()))
            raise TypeError("Got unexpected keyword argument ({})".format(a_key))

        self.pidgets = WeakKeyDictionary()

        doc = self.generate_doc().strip()
        if doc:
            self.__doc__ = doc

    def update_pidget(self, obj):
        pidget = self.get_pidget(obj)
        if pidget is not None:
            pidget.update()

    def get_pidget(self, obj):
        return self.pidgets.get(obj)

    def create_pidget(self, obj):
        if self.pidget_class is None:
            return None

        if obj not in self.pidgets:
            self.pidgets[obj] = self.pidget_class(self, obj)

        return self.pidgets[obj]

    def pidget_event_handler(self, *args, **kwargs):
        pass

    def dump(self):
        pass

    def load(self):
        pass

    def generate_doc(self):
        return self.help


class ValueParameter(Parameter):
    def __init__(self, **kwargs):
        self.default_value = kwargs.pop("default_value")
        self.override_check_fun = kwargs.pop("override_check_fun", None)

        kwargs.setdefault("does_dump", True)

        self.values = WeakKeyDictionary()

        super().__init__(**kwargs)

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self

        return copy(self.values.get(obj, default=self.check(obj, self.default_value)))

    def __set__(self, obj, value):
        value = self.check(obj, value)
        self.values[obj] = value

    def __delete__(self, obj):
        self.values.pop(obj, None)

    def pidget_event_handler(self, obj, val):
        self.__set__(obj, val)  # might raise an exception

        obj._parameter_event_handler()

    def check(self, obj, value):
        check_fun = self.override_check_fun or self._check
        return check_fun(obj, value)

    def _check(self, obj, value):
        return value

    def recheck(self, obj):
        self.check(obj, self.__get__(obj))

    def dump(self, obj):
        return self.__get__(obj)

    def load(self, obj, value):
        self.__set__(obj, value)

    def generate_doc(self):
        s = ""
        if self.help:
            s += self.help
            s += "\n\n"

        type_str = getattr(self, "type_str", None)
        if type_str:
            s += "| Type: {}\n".format(type_str)

        unit = getattr(self, "unit", None)
        if unit:
            s += "| Unit: {}\n".format(unit)

        s += "| Default value: {}\n".format(self.default_value)

        return s


class BoolParameter(ValueParameter):
    type_str = "bool"

    def __init__(self, **kwargs):
        kwargs.setdefault("pidget_class", BoolCheckboxPidget)

        super().__init__(**kwargs)

    def _check(self, obj, value):
        return bool(value)


class EnumParameter(ValueParameter):
    def __init__(self, **kwargs):
        self.enum = kwargs.pop("enum")

        kwargs.setdefault("pidget_class", ComboBoxPidget)

        super().__init__(**kwargs)

    def dump(self, obj):
        return self.__get__(obj).name

    def load(self, obj, value):
        value = self.enum[value]
        self.__set__(obj, value)


class NumberParameter(ValueParameter):
    def __init__(self, **kwargs):
        self.unit = kwargs.pop("unit", None)
        self.limits = kwargs.pop("limits", None)

        super().__init__(**kwargs)


class IntParameter(NumberParameter):
    type_str = "int"

    def __init__(self, **kwargs):
        self.step = kwargs.pop("step", 1)

        kwargs.setdefault("pidget_class", IntSpinBoxPidget)

        super().__init__(**kwargs)

    def _check(self, obj, value):
        if isinstance(value, float):
            raise ValueError("Not an integer")

        try:
            value = int(value)
        except ValueError:
            raise ValueError("Not a valid integer")

        if self.limits is not None:
            lower, upper = self.limits

            if value < lower:
                raise ValueError("Given value is too low")
            if value > upper:
                raise ValueError("Given value is too high")

        return value


class FloatParameter(NumberParameter):
    type_str = "float"

    def __init__(self, **kwargs):
        self.decimals = kwargs.pop("decimals", 2)
        self.logscale = kwargs.pop("logscale", False)

        kwargs.setdefault("pidget_class", FloatSpinBoxAndSliderPidget)

        super().__init__(**kwargs)

        if self.logscale:
            assert self.limits is not None
            assert self.limits[1] > self.limits[0] > 0

    def _check(self, obj, value):
        try:
            value = float(value)
        except ValueError:
            raise ValueError("Not a valid number")

        value = round(value, self.decimals)

        if self.limits is not None:
            lower, upper = self.limits

            if value < lower:
                raise ValueError("Given value is too low")
            if value > upper:
                raise ValueError("Given value is too high")

        return value


class FloatRangeParameter(FloatParameter):
    def __init__(self, **kwargs):
        kwargs.setdefault("pidget_class", FloatRangeSpinBoxesPidget)

        super().__init__(**kwargs)

    def _check(self, obj, arg):
        try:
            values = list(arg)
        except (ValueError, TypeError):
            raise ValueError("Not a valid range")

        if len(values) != 2:
            raise ValueError("Given range does not have two values")

        for i in range(2):
            values[i] = super()._check(obj, values[i])

        if values[0] > values[1]:
            raise ValueError("Invalid range")

        return values


class ClassParameter(Parameter):
    def __init__(self, **kwargs):
        self.objtype = kwargs.pop("objtype")

        self.instances = WeakKeyDictionary()

        super().__init__(**kwargs)

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self

        if obj not in self.instances:
            self.instances[obj] = self.objtype(self, obj)

        return self.instances[obj]

    def __set__(self, obj, value):
        raise AttributeError("Unsettable parameter")

    def __delete__(self, obj):
        self.instances.pop(obj, None)


def get_virtual_parameter_class(base_class):
    assert issubclass(base_class, ValueParameter)

    class VirtualParameter(base_class):
        def __init__(self, **kwargs):
            self.get_fun = kwargs.pop("get_fun")
            self.set_fun = kwargs.pop("set_fun", None)

            kwargs.setdefault("does_dump", False)

            kwargs.setdefault("default_value", None)

            super().__init__(**kwargs)

        def __get__(self, obj, objtype=None):
            if obj is None:
                return self

            return self.get_fun(obj)

        def __set__(self, obj, value):
            if self.set_fun is None:
                raise AttributeError("Unsettable parameter")

            self.set_fun(obj, value)

        def __delete__(self, obj):
            pass

    return VirtualParameter


class Config:
    class State(Enum):
        UNLOADED = 1
        LOADED = 2
        LIVE = 3

    VERSION = None

    _event_handlers = None
    __state = None

    def __init__(self):
        self._event_handlers = set()
        self.__state = Config.State.UNLOADED

    def _loads(self, s):
        d = json.loads(s)

        version = d.pop("VERSION", None)
        if version != self.VERSION:
            raise ValueError("Configuration version mismatch")

        params = dict(self._get_keys_and_params())
        for k, v in d.items():
            params[k].load(self, v)

        self._update_pidgets()

    def _dumps(self):
        d = {k: p.dump(self) for k, p in self._get_keys_and_params() if p.does_dump}

        if self.VERSION is not None:
            d["VERSION"] = self.VERSION

        return json.dumps(d)

    def _reset(self):
        params = self._get_params()
        for param in params:
            param.__delete__(self)
            param.update_pidget(self)

        self._parameter_event_handler()

    def _create_pidgets(self):
        params = self._get_params()
        pidgets = [param.create_pidget(self) for param in params]
        return pidgets

    def _update_pidgets(self):
        params = self._get_params()
        for param in params:
            param.update_pidget(self)

    def _parameter_event_handler(self):
        for event_handler in self._event_handlers:
            event_handler(self)

    def _get_keys_and_params(self):
        keys = dir(self)
        attrs = [getattr(type(self), key, None) for key in keys]
        z = [(k, a) for k, a in zip(keys, attrs) if isinstance(a, Parameter)]
        return sorted(z, key=lambda t: t[1].order)

    def _get_params(self):
        return [a for k, a in self._get_keys_and_params()]

    @property
    def _is_valid(self):
        for param in self._get_params():
            try:
                param.recheck(self)
            except AttributeError:
                pass
            except ValueError:
                return False

        return True

    @property
    def _state(self):
        return self.__state

    @_state.setter
    def _state(self, state):
        self.__state = state
        self._update_pidgets()

    def __setattr__(self, name, value):
        if hasattr(self, name):
            object.__setattr__(self, name, value)
        else:
            fmt = "'{}' object has no attribute '{}'"
            raise AttributeError(fmt.format(self.__class__.__name__, name))


class ReferenceData:
    def __init__(self, param, parent_instance, buffer_size=50):
        self._param = param
        self._parent_instance = parent_instance

        self._buffer_size = buffer_size  # TODO: make settable from parameter

        self._buffered_data = None
        self._loaded_data = None
        self._use = True
        self._source = None
        self._source_file = None
        self._error = None

    @property
    def buffer_size(self):
        return self._buffer_size

    @buffer_size.setter
    def buffer_size(self, value):
        self._buffer_size = value
        self.__notify()

    @property
    def buffered_data(self):
        return self._buffered_data

    @buffered_data.setter
    def buffered_data(self, value):
        self._buffered_data = value
        self.__notify()

    @property
    def loaded_data(self):
        return self._loaded_data

    @loaded_data.setter
    def loaded_data(self, value):
        self._loaded_data = value
        self.__notify()

    @property
    def use(self):
        return self._use

    @use.setter
    def use(self, value):
        self._use = value
        self.__notify()

    @property
    def source(self):
        return self._source

    @source.setter
    def source(self, value):
        self._source = value
        self.__notify()

    @property
    def source_file(self):
        return self._source_file

    @source_file.setter
    def source_file(self, value):
        self._source_file = value
        self.__notify()

    @property
    def error(self):
        return self._error

    @error.setter
    def error(self, value):
        self._error = value
        self.__notify()

    @property
    def is_loaded(self):
        return self.loaded_data is not None

    @property
    def has_buffered(self):
        return self.buffered_data is not None

    def load_buffered(self):
        self.loaded_data = self.buffered_data
        self.use = True
        self.source = "buffer"
        self.source_file = None

    def unload(self):
        self.loaded_data = None
        self.source = None
        self.source_file = None
        self.error = None

    def load_from_file(self, filename):
        self.loaded_data = np.load(filename, allow_pickle=True)
        self.use = True
        self.source = "file"
        self.source_file = os.path.basename(filename)

    def save_to_file(self, filename):
        np.save(filename, self.loaded_data, allow_pickle=True)
        self.source = "file"
        self.source_file = os.path.basename(filename)

    def __notify(self):
        self._parent_instance._parameter_event_handler()


class ProcessingConfig(Config):
    pass


class ReferenceDataPidget(PidgetStub):
    def __init__(self, param, parent_instance):
        super().__init__(param, parent_instance)

        self.title_label = QLabel(param.label)
        self.grid.addWidget(self.title_label, 0, 0, 1, 2)

        self.buffer_btn = QPushButton()
        self.buffer_btn.clicked.connect(self.__buffer_btn_clicked)
        self.grid.addWidget(self.buffer_btn, 1, 0, 1, 1)

        self.load_btn = QPushButton()
        self.load_btn.clicked.connect(self.__load_btn_clicked)
        self.grid.addWidget(self.load_btn, 1, 1, 1, 1)

        self.info_label = QLabel()
        self.grid.addWidget(self.info_label, 2, 0, 1, 2)

        self.use_cb = QCheckBox("Enable", self)
        self.use_cb.setTristate(False)
        self.use_cb.stateChanged.connect(self.__use_cb_state_changed)
        self.grid.addWidget(self.use_cb, 3, 0, 1, 2)

        self.update()

    def _update(self):
        super()._update()

        config = self._get_param_value()

        if config.is_loaded:
            enabled = config.source == "buffer"
            text = "Save" if enabled else "Already saved"
        else:
            enabled = config.has_buffered

            if config.has_buffered:
                text = "Use measured"
            else:
                if self._parent_instance._state == Config.State.LIVE:
                    text = "Measuring..."
                else:
                    text = "Waiting..."

        self.buffer_btn.setEnabled(enabled)
        self.buffer_btn.setText(text)

        text = "Unload" if config.is_loaded else "Load from file"
        self.load_btn.setText(text)

        if config.is_loaded:
            if config.source == "file":
                text = "Loaded {}".format(config.source_file)
            else:
                text = "Loaded measured data"
        else:
            text = ""

        self.info_label.setText(text)
        self.info_label.setVisible(config.is_loaded)

        self.use_cb.setEnabled(not config.error)
        self.use_cb.setChecked(False if config.error else config.use)
        self.use_cb.setVisible(config.is_loaded)

        if config.error:
            self._set_error(config.error)
        else:
            self._unset_error()

    def __buffer_btn_clicked(self):
        config = self._get_param_value()

        if config.is_loaded:
            filename = self.__file_dialog(save=True)
            if filename:
                config.save_to_file(filename)
        else:
            config.load_buffered()

    def __load_btn_clicked(self):
        config = self._get_param_value()

        if config.is_loaded:
            config.unload()
        else:
            filename = self.__file_dialog(save=False)
            if filename:
                config.load_from_file(filename)

    def __use_cb_state_changed(self, state):
        config = self._get_param_value()
        config.use = bool(state)

    def __file_dialog(self, save=True):
        caption = "Save" if save else "Load"
        caption = caption + " " + self.param.label.lower().strip()
        suggested_filename = self.param.label.lower().strip().replace(" ", "_") + ".npy"
        file_types = "NumPy data files (*.npy)"

        if save:
            filename, info = QFileDialog.getSaveFileName(
                    self,
                    caption,
                    suggested_filename,
                    file_types,
                    )
        else:
            filename, info = QFileDialog.getOpenFileName(
                    self,
                    caption,
                    suggested_filename,
                    file_types,
                    )

        return filename


class ReferenceDataParameter(ClassParameter):
    def __init__(self, **kwargs):
        kwargs.setdefault("label", "Background")
        kwargs.setdefault("pidget_class", ReferenceDataPidget)
        kwargs.setdefault("objtype", ReferenceData)
        kwargs.setdefault("updateable", True)

        super().__init__(**kwargs)
