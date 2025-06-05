using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Tilemaps;

namespace UnityEditor.Tilemaps
{
    /// <summary>
    ///     This Brush helps to pick Tiles which are grouped together by position. Gaps can be set to identify if Tiles belong
    ///     to a Group. Limits can be set to ensure that an over-sized Group will not be picked. Use this as an example to
    ///     create brushes that have the ability to choose and pick whichever Tiles it is interested in.
    /// </summary>
    [HelpURL(
        "https://docs.unity3d.com/Packages/com.unity.2d.tilemap.extras@latest/index.html?subfolder=/manual/GroupBrush.html")]
    [CustomGridBrush(true, false, false, "Group Brush")]
    public class GroupBrush : GridBrush
    {
        [SerializeField] private Vector3Int m_Gap = Vector3Int.one;

        [SerializeField] private Vector3Int m_Limit = Vector3Int.one * 3;

        [SerializeField] private readonly Stack<Vector3Int> m_NextPosition = new();

        [SerializeField] private BitArray m_VisitedLocations = new(7 * 7 * 7);

        /// <summary>
        ///     The gap in cell count before stopping to consider a Tile in a Group
        /// </summary>
        public Vector3Int gap
        {
            get => m_Gap;
            set
            {
                m_Gap = value;
                OnValidate();
            }
        }

        /// <summary>
        ///     The count in cells beyond the initial position before stopping to consider a Tile in a Group
        /// </summary>
        public Vector3Int limit
        {
            get => m_Limit;
            set
            {
                m_Limit = value;
                OnValidate();
            }
        }

        private int visitedLocationsSize => (m_Limit.x * 2 + 1) * (m_Limit.y * 2 + 1) * (m_Limit.z * 2 + 1);

        private void OnValidate()
        {
            if (m_Gap.x < 0)
                m_Gap.x = 0;
            if (m_Gap.y < 0)
                m_Gap.y = 0;
            if (m_Gap.z < 0)
                m_Gap.z = 0;
            if (m_Limit.x < 0)
                m_Limit.x = 0;
            if (m_Limit.y < 0)
                m_Limit.y = 0;
            if (m_Limit.z < 0)
                m_Limit.z = 0;
            if (m_VisitedLocations.Length != visitedLocationsSize)
                m_VisitedLocations = new BitArray(visitedLocationsSize);
        }

        /// <summary>
        ///     Picks tiles from selected Tilemaps and child GameObjects, given the coordinates of the cells.
        ///     The GroupBrush overrides this to locate groups of Tiles from the picking position.
        /// </summary>
        /// <param name="grid">Grid to pick data from.</param>
        /// <param name="brushTarget">Target of the picking operation. By default the currently selected GameObject.</param>
        /// <param name="position">The coordinates of the cells to paint data from.</param>
        /// <param name="pickStart">Pivot of the picking brush.</param>
        public override void Pick(GridLayout grid, GameObject brushTarget, BoundsInt position, Vector3Int pickStart)
        {
            // Do standard pick if user has selected a custom bounds
            if (position.size.x > 1 || position.size.y > 1 || position.size.z > 1)
            {
                base.Pick(grid, brushTarget, position, pickStart);
                return;
            }

            var tilemap = brushTarget.GetComponent<Tilemap>();
            if (tilemap == null)
                return;

            Reset();

            // Determine size of picked locations based on gap and limit
            var limitOrigin = position.position - limit;
            var limitSize = Vector3Int.one + limit * 2;
            var limitBounds = new BoundsInt(limitOrigin, limitSize);
            var pickBounds = new BoundsInt(position.position, Vector3Int.one);

            m_VisitedLocations.SetAll(false);
            m_VisitedLocations.Set(GetIndex(position.position, limitOrigin, limitSize), true);
            m_NextPosition.Clear();
            m_NextPosition.Push(position.position);

            while (m_NextPosition.Count > 0)
            {
                var next = m_NextPosition.Pop();
                if (tilemap.GetTile(next) != null)
                {
                    Encapsulate(ref pickBounds, next);
                    var gapBounds = new BoundsInt(next - gap, Vector3Int.one + gap * 2);
                    foreach (var gapPosition in gapBounds.allPositionsWithin)
                    {
                        if (!limitBounds.Contains(gapPosition))
                            continue;
                        var index = GetIndex(gapPosition, limitOrigin, limitSize);
                        if (!m_VisitedLocations.Get(index))
                        {
                            m_NextPosition.Push(gapPosition);
                            m_VisitedLocations.Set(index, true);
                        }
                    }
                }
            }

            UpdateSizeAndPivot(pickBounds.size, position.position - pickBounds.position);

            foreach (var pos in pickBounds.allPositionsWithin)
            {
                var brushPosition = new Vector3Int(pos.x - pickBounds.x, pos.y - pickBounds.y, pos.z - pickBounds.z);
                if (m_VisitedLocations.Get(GetIndex(pos, limitOrigin, limitSize)))
                    PickCell(pos, brushPosition, tilemap);
            }
        }

        private void Encapsulate(ref BoundsInt bounds, Vector3Int position)
        {
            if (bounds.Contains(position))
                return;

            if (position.x < bounds.position.x)
            {
                var increase = bounds.x - position.x;
                bounds.position = new Vector3Int(position.x, bounds.y, bounds.z);
                bounds.size = new Vector3Int(bounds.size.x + increase, bounds.size.y, bounds.size.z);
            }

            if (position.x >= bounds.xMax)
            {
                var increase = position.x - bounds.xMax + 1;
                bounds.size = new Vector3Int(bounds.size.x + increase, bounds.size.y, bounds.size.z);
            }

            if (position.y < bounds.position.y)
            {
                var increase = bounds.y - position.y;
                bounds.position = new Vector3Int(bounds.x, position.y, bounds.z);
                bounds.size = new Vector3Int(bounds.size.x, bounds.size.y + increase, bounds.size.z);
            }

            if (position.y >= bounds.yMax)
            {
                var increase = position.y - bounds.yMax + 1;
                bounds.size = new Vector3Int(bounds.size.x, bounds.size.y + increase, bounds.size.z);
            }

            if (position.z < bounds.position.z)
            {
                var increase = bounds.z - position.z;
                bounds.position = new Vector3Int(bounds.x, bounds.y, position.z);
                bounds.size = new Vector3Int(bounds.size.x, bounds.size.y, bounds.size.z + increase);
            }

            if (position.z >= bounds.zMax)
            {
                var increase = position.z - bounds.zMax + 1;
                bounds.size = new Vector3Int(bounds.size.x, bounds.size.y, bounds.size.z + increase);
            }
        }

        private int GetIndex(Vector3Int position, Vector3Int origin, Vector3Int size)
        {
            return (position.z - origin.z) * size.y * size.x
                   + (position.y - origin.y) * size.x
                   + (position.x - origin.x);
        }

        private void PickCell(Vector3Int position, Vector3Int brushPosition, Tilemap tilemap)
        {
            if (tilemap != null)
            {
                SetTile(brushPosition, tilemap.GetTile(position));
                SetMatrix(brushPosition, tilemap.GetTransformMatrix(position));
                SetColor(brushPosition, tilemap.GetColor(position));
            }
        }
    }

    /// <summary>
    ///     The Brush Editor for a Group Brush.
    /// </summary>
    [CustomEditor(typeof(GroupBrush))]
    public class GroupBrushEditor : GridBrushEditor
    {
        private static readonly string iconPath =
            "Packages/com.unity.2d.tilemap.extras/Editor/Brushes/GroupBrush/GroupBrush.png";

        private Texture2D m_BrushIcon;

        /// <summary> Returns an icon identifying the Group Brush. </summary>
        public override Texture2D icon
        {
            get
            {
                if (m_BrushIcon == null)
                {
                    var gui = EditorGUIUtility.TrIconContent(iconPath);
                    m_BrushIcon = gui.image as Texture2D;
                }

                return m_BrushIcon;
            }
        }
    }
}