using System;
using System.Collections.Generic;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;

namespace UnityEditor.U2D.Aseprite
{
    [Serializable]
    internal class Layer
    {
        [SerializeField] int m_LayerIndex;
        [SerializeField] UUID m_Uuid;
        [SerializeField] int m_Guid; // Only used if the Aseprite file does not contain a UUID.
        [SerializeField] string m_Name;
        [SerializeField] LayerFlags m_LayerFlags;
        [SerializeField] LayerTypes m_LayerType;
        [SerializeField] BlendModes m_BlendMode;
        [SerializeField] List<Cell> m_Cells = new ();
        [SerializeField] List<LinkedCell> m_LinkedCells = new ();
        [SerializeField] List<TileCell> m_TileCells = new ();
        [SerializeField] uint m_TileSetIndex;
        [SerializeField] int m_ParentIndex = -1;

        [NonSerialized] public float opacity;

        public int index
        {
            get => m_LayerIndex;
            set => m_LayerIndex = value;
        }
        /// <summary>
        /// The UUID is a Universally Unique Identifier used to identify the layer.
        /// The UUID is either read from the Aseprite file (if available), or generated from the layer name and its parent's name.
        /// </summary>
        public UUID uuid
        {
            get => m_Uuid;
            set => m_Uuid = value;
        }
        /// <summary>
        /// The GUID is generated from the layer name and its parent's name.
        /// The GUID is being deprecated in favor of the UUID.
        /// </summary>
        public int guid
        {
            get => m_Guid;
            set => m_Guid = value;
        }
        public string name
        {
            get => m_Name;
            set => m_Name = value;
        }
        public LayerFlags layerFlags
        {
            get => m_LayerFlags;
            set => m_LayerFlags = value;
        }
        public LayerTypes layerType
        {
            get => m_LayerType;
            set => m_LayerType = value;
        }
        public BlendModes blendMode
        {
            get => m_BlendMode;
            set => m_BlendMode = value;
        }
        public List<Cell> cells
        {
            get => m_Cells;
            set => m_Cells = value;
        }
        public List<LinkedCell> linkedCells
        {
            get => m_LinkedCells;
            set => m_LinkedCells = value;
        }
        public List<TileCell> tileCells
        {
            get => m_TileCells;
            set => m_TileCells = value;
        }
        public uint tileSetIndex
        {
            get => m_TileSetIndex;
            set => m_TileSetIndex = value;
        }        
        public int parentIndex
        {
            get => m_ParentIndex;
            set => m_ParentIndex = value;
        }

        public static int GenerateGuid(Layer layer, IReadOnlyList<Layer> layers)
        {
            var fullName = layer.name;
            var parent = layer;
            do
            {
                var parentIndex = parent.parentIndex;
                parent = layers.Find(x => x.index == parentIndex);
                if (parent != null)
                    fullName = fullName.Insert(0, parent.name + "/");
                
            } while (parent != null);
            
            var hash = fullName.GetHashCode();
            return hash;
        }
    }

    [Serializable]
    internal struct Cell : IEquatable<Cell>
    {
        [SerializeField] string m_Name;
        [SerializeField] int m_FrameIndex;
        [SerializeField] int m_AdditiveSortOrder;
        [SerializeField] RectInt m_CellRect;
        [SerializeField] string m_SpriteId;

        [NonSerialized] public bool updatedCellRect;
        [NonSerialized] public float opacity;
        [NonSerialized] public BlendModes blendMode;
        [NonSerialized] public NativeArray<Color32> image;

        public string name
        {
            get => m_Name;
            set => m_Name = value;
        }
        public int frameIndex
        {
            get => m_FrameIndex;
            set => m_FrameIndex = value;
        }
        public int additiveSortOrder
        {
            get => m_AdditiveSortOrder;
            set => m_AdditiveSortOrder = value;
        }
        public RectInt cellRect
        {
            get => m_CellRect;
            set => m_CellRect = value;
        }
        public GUID spriteId
        {
            get => new GUID(m_SpriteId);
            set => m_SpriteId = value.ToString();
        }

        public bool Equals(Cell other)
        {
            return m_Name == other.m_Name && 
                   m_FrameIndex == other.m_FrameIndex && 
                   m_AdditiveSortOrder == other.m_AdditiveSortOrder && 
                   m_CellRect.Equals(other.m_CellRect) && 
                   m_SpriteId == other.m_SpriteId && 
                   updatedCellRect == other.updatedCellRect && 
                   opacity.Equals(other.opacity) && 
                   blendMode == other.blendMode && 
                   image.Equals(other.image);
        }

        public override bool Equals(object obj)
        {
            return obj is Cell other && Equals(other);
        }

        public override int GetHashCode()
        {
            unchecked
            {
                var hashCode = (m_Name != null ? m_Name.GetHashCode() : 0);
                hashCode = (hashCode * 397) ^ m_FrameIndex;
                hashCode = (hashCode * 397) ^ m_AdditiveSortOrder;
                hashCode = (hashCode * 397) ^ m_CellRect.GetHashCode();
                hashCode = (hashCode * 397) ^ (m_SpriteId != null ? m_SpriteId.GetHashCode() : 0);
                hashCode = (hashCode * 397) ^ updatedCellRect.GetHashCode();
                hashCode = (hashCode * 397) ^ opacity.GetHashCode();
                hashCode = (hashCode * 397) ^ (int) blendMode;
                hashCode = (hashCode * 397) ^ image.GetHashCode();
                return hashCode;
            }
        }
    }

    [Serializable]
    internal class LinkedCell
    {
        [SerializeField] int m_FrameIndex;
        [SerializeField] int m_LinkedToFrame;

        public int frameIndex
        {
            get => m_FrameIndex;
            set => m_FrameIndex = value;
        }
        public int linkedToFrame
        {
            get => m_LinkedToFrame;
            set => m_LinkedToFrame = value;
        }
    }

    [Serializable]
    internal struct TileCell
    {
        [SerializeField] uint m_LayerIndex;
        [SerializeField] int m_FrameIndex;
        [SerializeField] RectInt m_CellRect;
        [SerializeField] uint[] m_TileIndices;

        public uint layerIndex
        {
            get => m_LayerIndex;
            set => m_LayerIndex = value;
        }
        public int frameIndex
        {
            get => m_FrameIndex;
            set => m_FrameIndex = value;
        }
        public RectInt cellRect
        {
            get => m_CellRect;
            set => m_CellRect = value;
        }
        public uint[] tileIndices
        {
            get => m_TileIndices;
            set => m_TileIndices = value;
        }
    }

    internal class Frame
    {
        int m_Duration;
        (string, object)[] m_EventData;

        public int duration
        {
            get => m_Duration;
            set => m_Duration = value;
        }
        public (string, object)[] eventData
        {
            get => m_EventData;
            set => m_EventData = value;
        }
    }

    internal class Tag
    {
        public string name { get; set; }
        public int fromFrame { get; set; }
        public int toFrame { get; set; }
        public int noOfRepeats { get; set; }

        public int noOfFrames => toFrame - fromFrame;
        public bool isRepeating => noOfRepeats == 0;
    }

    [Serializable]
    internal class TileSet
    {
        [SerializeField] uint m_Id;
        [SerializeField] string m_Name;
        [SerializeField] int2 m_TileSize;
        [SerializeField] List<Tile> m_Tiles = new ();
        [SerializeField] int m_Guid;

        public uint id
        {
            get => m_Id;
            set => m_Id = value;
        }
        
        public string name
        {
            get => m_Name;
            set => m_Name = value;
        }
        public int2 tileSize
        {
            get => m_TileSize;
            set => m_TileSize = value;
        }
        public List<Tile> tiles
        {
            get => m_Tiles;
            set => m_Tiles = value;
        }
        public int guid
        {
            get => m_Guid;
            set => m_Guid = value;
        }
        
        public static int GenerateGuid(TileSet tileSet)
        {
            var hash = tileSet.name.GetHashCode();
            hash = (hash * 397) ^ tileSet.id.GetHashCode();
            return hash;
        }        
    }

    [Serializable]
    internal struct Tile
    {
        [NonSerialized] public NativeArray<Color32> image;
        [SerializeField] uint m_TileId;
        [SerializeField] int2 m_Size;
        [SerializeField] string m_Name;
        [SerializeField] string m_SpriteId;

        public uint tileId
        {
            get => m_TileId;
            set => m_TileId = value;
        }
        public string name
        {
            get => m_Name;
            set => m_Name = value;
        }
        public int2 size
        {
            get => m_Size;
            set => m_Size = value;
        }
        public GUID spriteId
        {
            get => new GUID(m_SpriteId);
            set => m_SpriteId = value.ToString();
        }
    }

    /// <summary>
    /// Import modes for the file.
    /// </summary>
    public enum FileImportModes
    {
        /// <summary>The file is imported as a Sprite Sheet, and can be sliced up in the Sprite Editor.</summary>
        SpriteSheet = 0,
        /// <summary>The file is imported with animation in mind. Animation assets are generated and attached to a model prefab on import.</summary>
        AnimatedSprite = 1,
        /// <summary>he importer finds all tile data in the file and generates Unity Tilemap assets on import.</summary>
        TileSet = 2
    }

    /// <summary>
    /// Import modes for all layers.
    /// </summary>
    public enum LayerImportModes
    {
        /// <summary>
        /// Every layer per frame generates a Sprite.
        /// </summary>
        IndividualLayers,
        /// <summary>
        /// All layers per frame are merged into one Sprite.
        /// </summary>
        MergeFrame
    }

    /// <summary>
    /// The space the Sprite pivots are being calculated.
    /// </summary>
    public enum PivotSpaces
    {
        /// <summary>
        /// Canvas space. Calculate the pivot based on where the Sprite is positioned on the source asset's canvas.
        /// This is useful if the Sprite is being swapped out in an animation.
        /// </summary>
        Canvas,
        /// <summary>
        /// Local space. This is the normal pivot space.
        /// </summary>
        Local
    }

    /// <summary>
    /// Universally Unique Identifier used by Aseprite
    /// </summary>
    [Serializable]
    internal struct UUID : IComparable, IComparable<UUID>, IEquatable<UUID>
    {
        public static readonly UUID zero = default;
        
        [SerializeField] uint m_Value0;
        [SerializeField] uint m_Value1;
        [SerializeField] uint m_Value2;
        [SerializeField] uint m_Value3;

        public UUID(uint value0, uint value1, uint value2, uint value3)
        {
            m_Value0 = value0;
            m_Value1 = value1;
            m_Value2 = value2;
            m_Value3 = value3;
        }
        
        public static bool operator==(UUID x, UUID y)
        {
            return x.m_Value0 == y.m_Value0 && x.m_Value1 == y.m_Value1 && x.m_Value2 == y.m_Value2 && x.m_Value3 == y.m_Value3;
        }

        public static bool operator!=(UUID x, UUID y)
        {
            return !(x == y);
        }        
        
        public static bool operator<(UUID x, UUID y)
        {
            if (x.m_Value0 != y.m_Value0)
                return x.m_Value0 < y.m_Value0;
            if (x.m_Value1 != y.m_Value1)
                return x.m_Value1 < y.m_Value1;
            if (x.m_Value2 != y.m_Value2)
                return x.m_Value2 < y.m_Value2;
            return x.m_Value3 < y.m_Value3;
        }

        public static bool operator>(UUID x, UUID y)
        {
            if (x < y)
                return false;
            if (x == y)
                return false;
            return true;
        }        

        public override bool Equals(object obj)
        {
            return obj is UUID uuid && Equals(uuid);
        }

        public bool Equals(UUID obj)
        {
            return this == obj;
        }

        public override int GetHashCode()
        {
            unchecked
            {
                var hashCode = (int)m_Value0;
                hashCode = (hashCode * 397) ^ (int)m_Value1;
                hashCode = (hashCode * 397) ^ (int)m_Value2;
                hashCode = (hashCode * 397) ^ (int)m_Value3;
                return hashCode;
            }
        }

        public int CompareTo(object obj)
        {
            if (obj == null)
                return 1;

            var rhs = (UUID)obj;
            return this.CompareTo(rhs);
        }

        public int CompareTo(UUID rhs)
        {
            if (this < rhs)
                return -1;
            if (this > rhs)
                return 1;
            return 0;
        }

        public override string ToString()
        {
            return $"{m_Value0:X8}-{m_Value1:X8}-{m_Value2:X8}-{m_Value3:X8}";
        }
    }
}
