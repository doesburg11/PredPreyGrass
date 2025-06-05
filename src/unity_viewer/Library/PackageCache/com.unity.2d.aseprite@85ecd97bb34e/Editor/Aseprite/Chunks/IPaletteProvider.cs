using System.Collections.ObjectModel;

namespace UnityEditor.U2D.Aseprite
{
    internal interface IPaletteProvider
    {
        public ReadOnlyCollection<PaletteEntry> entries { get; }
    }
}
