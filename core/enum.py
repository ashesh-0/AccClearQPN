class Enum:
    @classmethod
    def name(cls, enum_type):
        for key, value in cls.__dict__.items():
            if enum_type == value:
                return key

    @classmethod
    def from_name(cls, enum_type_str):
        for key, value in cls.__dict__.items():
            if key == enum_type_str:
                return value
        assert f'{cls.__name__}:{enum_type_str} doesnot exist.'


class DataType:
    # NOTE: Every value has to be a power of 2.
    NoneAtAll = 0
    Rain = 1
    RainDiff = 2
    Hour = 4
    Month = 8
    Radar = 16
    Altitude = 32
    Latitude = 64
    Longitude = 128

    @classmethod
    def all_data(cls):
        output = 0
        for key, value in cls.__dict__.items():
            if not isinstance(value, int):
                continue
            output += value
        return output

    @classmethod
    def count(cls, dtype):
        return cls.count2D(dtype) + cls.count1D(dtype)

    @classmethod
    def count1D(cls, dtype):
        hour = int((dtype & cls.Hour) == cls.Hour)
        month = int((dtype & cls.Month) == cls.Month)
        return hour + month

    @classmethod
    def count2D(cls, dtype):
        rain = int(dtype & DataType.Rain == DataType.Rain)
        raindiff = int(dtype & DataType.RainDiff == DataType.RainDiff)
        radar = int(dtype & DataType.Radar == DataType.Radar)
        altitude = int(dtype & DataType.Altitude == DataType.Altitude)
        lat = int(dtype & DataType.Latitude == DataType.Latitude)
        lon = int(dtype & DataType.Longitude == DataType.Longitude)
        return rain + raindiff + radar + altitude + lat + lon

    @classmethod
    def print(cls, dtype, prefix=''):
        for key, value in cls.__dict__.items():
            if value == cls.NoneAtAll or not isinstance(value, int):
                continue
            if (dtype & value) == value:
                print(f'[{prefix} Dtype]', key)
