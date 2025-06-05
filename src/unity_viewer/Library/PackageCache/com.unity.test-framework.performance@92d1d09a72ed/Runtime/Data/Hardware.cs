using System;

namespace Unity.PerformanceTesting.Data
{
    /// <summary>
    /// Represents information about Hardware.
    /// </summary>
    [Serializable]
    public class Hardware
    {
        /// <summary>
        /// Operating system.
        /// </summary>
        [RequiredMember] public string OperatingSystem;

        /// <summary>
        /// Device model.
        /// </summary>
        [RequiredMember] public string DeviceModel;

        /// <summary>
        /// Device name.
        /// </summary>
        [RequiredMember] public string DeviceName;

        /// <summary>
        /// Processor type name.
        /// </summary>
        [RequiredMember] public string ProcessorType;

        /// <summary>
        /// Processor thread count.
        /// </summary>
        [RequiredMember] public int ProcessorCount;

        /// <summary>
        /// Graphics device name.
        /// </summary>
        [RequiredMember] public string GraphicsDeviceName;

        /// <summary>
        /// System Memory size in MegaBytes.
        /// </summary>
        [RequiredMember] public int SystemMemorySizeMB;
    }
}
