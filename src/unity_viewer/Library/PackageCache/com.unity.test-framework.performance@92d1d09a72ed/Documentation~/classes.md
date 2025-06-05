# Classes

This section contains a reference for classes relevant when working with the  Performance Testing Package.

## SampleGroup

**class SampleGroup** - represents a group of samples with the same purpose that share a name, sample unit and whether an increase is better. 

Optional parameters
- **Name** : Name of the measurement. If unspecified, "Time" is used as the default name.
- **Unit** : Unit of the measurement to report samples in. Possible values are:
Nanosecond, Microsecond, Millisecond, Second, Byte, Kilobyte, Megabyte, Gigabyte
- **IncreaseIsBetter** : If true, an increase in the measurement value is considered a performance improvement (progression). If false, an increase is treated as a performance regression. False by default. 
