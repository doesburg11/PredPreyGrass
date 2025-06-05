using System;
using UnityEngine.Tilemaps;
using UnityEngine;
using System.Collections.Generic;
using UnityEngine.Scripting.APIUpdating;

namespace UnityEditor.Tilemaps
{
    /// <summary>
    /// Default built-in brush for painting or erasing tiles and/or gameobjects on a grid.
    /// </summary>
    /// <remarks>
    /// Default brush is meant for two things: Authoring GameObjects on a grid and authoring tiles on a grid. It can be used for both at the same time if necessary. Basically the default brush allows you to pick and paste tiles and GameObjects from one area to another.
    ///
    /// Tiles in a tilemap are considered to be active for brush editing if a GameObject with Tilemap is currently selected.
    /// GameObjects are considered to be active for brush editing if their parent GameObject is currently selected.
    ///
    /// For example: A new default brush with size of 2x3 is generated as a result of Scene view picking operation. The new instance contains six cells (GridBrush.BrushCell) and the cells are fed with tiles &amp; GameObjects from the picking area.
    ///
    /// Later when this brush is used in a painting operation, all the tiles and GameObjects contained in the cells are set/cloned into a new Scene position.
    ///
    /// When creating custom brushes, it is recommended to inherit GridBrushBase by default. Inheriting GridBrush is possible when similar functionality is required, but extending it has its limits compared to base class.
    ///
    /// It is also possible to replace the default GridBrush from the Tile Palette brush list completely by using the GridDefaultBrush attribute on one of your custom brushes and promote it to become a new default brush of the project. This is useful when higher level brush can operate as a default and protect designers from accidentally using the built-in one.
    /// </remarks>
    [MovedFrom(true, "UnityEditor", "UnityEditor")]
    public class GridBrush : GridBrushBase
    {
        [SerializeField]
        [HideInInspector]
        private BrushCell[] m_Cells;

        [SerializeField]
        [HideInInspector]
        private List<TileChangeData> m_TileChangeDataList;

        [SerializeField]
        [HideInInspector]
        private Vector3Int m_Size;

        [SerializeField]
        [HideInInspector]
        private Vector3Int m_Pivot;

        [SerializeField]
        [HideInInspector]
        private bool m_CanChangeZPosition;

        [SerializeField]
        private bool m_FloodFillContiguousOnly = true;

        [SerializeField]
        [HideInInspector]
        private GridLayout.CellLayout m_LastPickedLayout = GridLayout.CellLayout.Rectangle;

        [SerializeField]
        [HideInInspector]
        private GridLayout.CellSwizzle m_LastPickedCellSwizzle = GridLayout.CellSwizzle.XYZ;

        [SerializeField]
        [HideInInspector]
        private Vector3 m_LastPickedCellSize = Vector3.one;

        [SerializeField]
        [HideInInspector]
        private Vector3 m_LastPickedCellGap = Vector3.zero;

        private Vector3Int m_StoredSize;
        private Vector3Int m_StoredPivot;
        private BrushCell[] m_StoredCells;

        private static readonly Matrix4x4 s_Clockwise = new Matrix4x4(new Vector4(0f, -1f, 0f, 0f), new Vector4(1f, 0f, 0f, 0f), new Vector4(0f, 0f, 1f, 0f), new Vector4(0f, 0f, 0f, 1f));
        private static readonly Matrix4x4 s_CounterClockwise = new Matrix4x4(new Vector4(0f, 1f, 0f, 0f), new Vector4(-1f, 0f, 0f, 0f), new Vector4(0f, 0f, 1f, 0f), new Vector4(0f, 0f, 0f, 1f));
        private static readonly Matrix4x4 s_HexagonClockwise = new Matrix4x4(new Vector4(Mathf.Cos(Mathf.Deg2Rad * -60f), Mathf.Sin(Mathf.Deg2Rad *-60f), 0f, 0f), new Vector4(-Mathf.Sin(Mathf.Deg2Rad * -60f), Mathf.Cos(Mathf.Deg2Rad * -60f), 0f, 0f), new Vector4(0f, 0f, 1f, 0f), new Vector4(0f, 0f, 0f, 1f));
        private static readonly Matrix4x4 s_HexagonCounterClockwise = new Matrix4x4(new Vector4(Mathf.Cos(Mathf.Deg2Rad * 60f), Mathf.Sin(Mathf.Deg2Rad * 60f), 0f, 0f), new Vector4(-Mathf.Sin(Mathf.Deg2Rad * 60f), Mathf.Cos(Mathf.Deg2Rad * 60f), 0f, 0f), new Vector4(0f, 0f, 1f, 0f), new Vector4(0f, 0f, 0f, 1f));
        private static readonly Matrix4x4 s_180Rotate = new Matrix4x4(new Vector4(-1f, 0f, 0f, 0f), new Vector4(0f, -1f, 0f, 0f), new Vector4(0f, 0f, 1f, 0f), new Vector4(0f, 0f, 0f, 1f));

        /// <summary>Size of the brush in cells. </summary>
        public Vector3Int size { get { return m_Size; } set { m_Size = value; SizeUpdated(); } }
        /// <summary>Pivot of the brush. </summary>
        public Vector3Int pivot { get { return m_Pivot; } set { m_Pivot = value; } }
        /// <summary>All the brush cells the brush holds. </summary>
        public BrushCell[] cells { get { return m_Cells; } }
        /// <summary>Number of brush cells in the brush.</summary>
        public int cellCount { get { return m_Cells?.Length ?? 0; } }
        /// <summary>Whether the brush can change Z Position</summary>
        public bool canChangeZPosition
        {
            get => m_CanChangeZPosition;
            set => m_CanChangeZPosition = value;
        }

        /// <summary>
        /// Cell Layout of the Grid which the GridBrush last picked from.
        /// </summary>
        public GridLayout.CellLayout lastPickedCellLayout => m_LastPickedLayout;
        /// <summary>
        /// Cell Swizzle of the Grid which the GridBrush last picked from.
        /// </summary>
        public GridLayout.CellSwizzle lastPickedCellSwizzle => m_LastPickedCellSwizzle;
        /// <summary>
        /// Cell Size of the Grid which the GridBrush last picked from.
        /// </summary>
        public Vector3 lastPickedCellSize => m_LastPickedCellSize;
        /// <summary>
        /// Cell Gap of the Grid which the GridBrush last picked from.
        /// </summary>
        public Vector3 lastPickedCellGap => m_LastPickedCellGap;

        /// <summary>
        /// Default built-in brush for painting or erasing Tiles and/or GameObjects on a Grid.
        /// </summary>
        public GridBrush()
        {
            Init(Vector3Int.one, Vector3Int.zero);
            SizeUpdated();
        }

        /// <summary>Initializes the content of the GridBrush.</summary>
        /// <param name="newSize">Size of the GridBrush.</param>
        public void Init(Vector3Int newSize)
        {
            Init(newSize, Vector3Int.zero);
            SizeUpdated();
        }

        /// <summary>Initializes the content of the GridBrush.</summary>
        /// <param name="newSize">Size of the GridBrush.</param>
        /// <param name="newPivot">Pivot point of the GridBrush.</param>
        public void Init(Vector3Int newSize, Vector3Int newPivot)
        {
            m_Size = newSize;
            m_Pivot = newPivot;
            SizeUpdated();
        }

        /// <summary>Paints tiles and GameObjects into a given position within the selected layers.</summary>
        /// <param name="gridLayout">Grid used for layout.</param>
        /// <param name="brushTarget">Target of the paint operation. By default the currently selected GameObject.</param>
        /// <param name="position">The coordinates of the cell to paint data to.</param>
        public override void Paint(GridLayout gridLayout, GameObject brushTarget, Vector3Int position)
        {
            Vector3Int min = position - pivot;
            BoundsInt bounds = new BoundsInt(min, m_Size);
            BoxFill(gridLayout, brushTarget, bounds);
        }

        /// <summary>Erases tiles and GameObjects in a given position within the selected layers.</summary>
        /// <param name="gridLayout">Grid used for layout.</param>
        /// <param name="brushTarget">Target of the erase operation. By default the currently selected GameObject.</param>
        /// <param name="position">The coordinates of the cell to erase data from.</param>
        public override void Erase(GridLayout gridLayout, GameObject brushTarget, Vector3Int position)
        {
            Vector3Int min = position - pivot;
            BoundsInt bounds = new BoundsInt(min, m_Size);
            BoxErase(gridLayout, brushTarget, bounds);
        }

        /// <summary>Box fills tiles and GameObjects into given bounds within the selected layers.</summary>
        /// <param name="gridLayout">Grid to box fill data to.</param>
        /// <param name="brushTarget">Target of the box fill operation. By default the currently selected GameObject.</param>
        /// <param name="position">The bounds to box fill data into.</param>
        public override void BoxFill(GridLayout gridLayout, GameObject brushTarget, BoundsInt position)
        {
            if (brushTarget == null)
                return;

            var map = brushTarget.GetComponent<Tilemap>();
            if (map == null)
                return;

            var count = 0;
            var listSize = position.size.x * position.size.y * position.size.z;
            if (m_TileChangeDataList == null || m_TileChangeDataList.Capacity != listSize)
                m_TileChangeDataList = new List<TileChangeData>(listSize);
            m_TileChangeDataList.Clear();
            foreach (var location in position.allPositionsWithin)
            {
                var local = location - position.min;
                var cell = m_Cells[GetCellIndexWrapAround(local.x, local.y, local.z)];
                if (cell.tile == null)
                    continue;

                var tcd = new TileChangeData { position = location, tile = cell.tile, transform = cell.matrix, color = cell.color };
                m_TileChangeDataList.Add(tcd);
                count++;
            }
            // Duplicate empty slots in the list, as ExtractArrayFromList returns full list
            if (0 < count && count < listSize)
            {
                var tcd = m_TileChangeDataList[count - 1];
                for (int i = count; i < listSize; ++i)
                {
                    m_TileChangeDataList.Add(tcd);
                }
            }
            var tileChangeData = NoAllocHelpers.ExtractArrayFromList(m_TileChangeDataList);
            map.SetTiles(tileChangeData, false);
        }

        /// <summary>Erases tiles and GameObjects from given bounds within the selected layers.</summary>
        /// <param name="gridLayout">Grid to erase data from.</param>
        /// <param name="brushTarget">Target of the erase operation. By default the currently selected GameObject.</param>
        /// <param name="position">The bounds to erase data from.</param>
        public override void BoxErase(GridLayout gridLayout, GameObject brushTarget, BoundsInt position)
        {
            if (brushTarget == null)
                return;

            var map = brushTarget.GetComponent<Tilemap>();
            if (map == null)
                return;

            var identity = Matrix4x4.identity;
            var listSize = Math.Abs(position.size.x * position.size.y * position.size.z);
            if (m_TileChangeDataList == null || m_TileChangeDataList.Capacity != listSize)
                m_TileChangeDataList = new List<TileChangeData>(listSize);
            m_TileChangeDataList.Clear();
            foreach (var location in position.allPositionsWithin)
            {
                m_TileChangeDataList.Add(new TileChangeData { position = location, tile = null, transform = identity, color = Color.white });
            }
            var tileChangeData = NoAllocHelpers.ExtractArrayFromList(m_TileChangeDataList);
            map.SetTiles(tileChangeData, false);
        }

        /// <summary>Flood fills tiles and GameObjects starting from a given position within the selected layers.</summary>
        /// <param name="gridLayout">Grid used for layout.</param>
        /// <param name="brushTarget">Target of the flood fill operation. By default the currently selected GameObject.</param>
        /// <param name="position">Starting position of the flood fill.</param>
        public override void FloodFill(GridLayout gridLayout, GameObject brushTarget, Vector3Int position)
        {
            if (cellCount == 0)
                return;

            if (brushTarget == null)
                return;

            var map = brushTarget.GetComponent<Tilemap>();
            if (map == null)
                return;

            if (m_FloodFillContiguousOnly)
            {
                map.FloodFill(position, cells[0].tile);
            }
            else
            {
                var tile = map.GetTile(position);
                if (tile != null && tile != cells[0].tile)
                {
                    map.SwapTile(tile, cells[0].tile);
                }
                else
                {
                    map.FloodFill(position, cells[0].tile);
                }
            }
        }

        /// <summary>Rotates the brush by 90 degrees in the given direction.</summary>
        /// <param name="direction">Direction to rotate by.</param>
        /// <param name="layout">Cell Layout for rotating.</param>
        public override void Rotate(RotationDirection direction, GridLayout.CellLayout layout)
        {
            switch (layout)
            {
                case GridLayout.CellLayout.Hexagon:
                    RotateHexagon(direction);
                    break;
                case GridLayout.CellLayout.Isometric:
                case GridLayout.CellLayout.IsometricZAsY:
                case GridLayout.CellLayout.Rectangle:
                {
                    var oldSize = m_Size;
                    var oldCells = m_Cells.Clone() as BrushCell[];
                    size = new Vector3Int(oldSize.y, oldSize.x, oldSize.z);
                    var oldBounds = new BoundsInt(Vector3Int.zero, oldSize);

                    foreach (var oldPos in oldBounds.allPositionsWithin)
                    {
                        var newX = direction == RotationDirection.Clockwise ? oldPos.y : oldSize.y - oldPos.y - 1;
                        var newY = direction == RotationDirection.Clockwise ? oldSize.x - oldPos.x - 1 : oldPos.x;
                        var toIndex = GetCellIndex(newX, newY, oldPos.z);
                        var fromIndex = GetCellIndex(oldPos.x, oldPos.y, oldPos.z, oldSize.x, oldSize.y, oldSize.z);
                        m_Cells[toIndex] = oldCells[fromIndex];
                    }

                    var newPivotX = direction == RotationDirection.Clockwise ? pivot.y : oldSize.y - pivot.y - 1;
                    var newPivotY = direction == RotationDirection.Clockwise ? oldSize.x - pivot.x - 1 : pivot.x;
                    pivot = new Vector3Int(newPivotX, newPivotY, pivot.z);

                    var rotation = direction == RotationDirection.Clockwise ? s_Clockwise : s_CounterClockwise;
                    var counterRotation = direction != RotationDirection.Clockwise ? s_Clockwise : s_CounterClockwise;
                    foreach (BrushCell cell in m_Cells)
                    {
                        var oldMatrix = cell.matrix;
                        var counter = (oldMatrix.lossyScale.x < 0) ^ (oldMatrix.lossyScale.y < 0);
                        cell.matrix = oldMatrix * (counter ? counterRotation : rotation);
                    }
                }
                break;
            }
        }

        private static Vector3Int RotateHexagonPosition(RotationDirection direction, Vector3Int position)
        {
            var cube = HexagonToCube(position);
            var rotatedCube = Vector3Int.zero;
            if (RotationDirection.Clockwise == direction)
            {
                rotatedCube.x = -cube.y;
                rotatedCube.y = -cube.z;
                rotatedCube.z = -cube.x;
            }
            else
            {
                rotatedCube.x = -cube.z;
                rotatedCube.y = -cube.x;
                rotatedCube.z = -cube.y;
            }
            return CubeToHexagon(rotatedCube);
        }

        private void RotateHexagon(RotationDirection direction)
        {
            var oldCells = m_Cells.Clone() as BrushCell[];
            var oldPivot = new Vector3Int(pivot.x, pivot.y, pivot.z);
            var oldSize = new Vector3Int(size.x, size.y, size.z);
            var minSize = Vector3Int.zero;
            var maxSize = Vector3Int.zero;
            var oldBounds = new BoundsInt(Vector3Int.zero, oldSize);
            foreach (var oldPos in oldBounds.allPositionsWithin)
            {
                if (oldCells[GetCellIndex(oldPos.x, oldPos.y, oldPos.z, oldSize.x, oldSize.y, oldSize.z)].tile == null)
                    continue;
                var pos = RotateHexagonPosition(direction, oldPos - oldPivot);
                minSize.x = Mathf.Min(minSize.x, pos.x);
                minSize.y = Mathf.Min(minSize.y, pos.y);
                maxSize.x = Mathf.Max(maxSize.x, pos.x);
                maxSize.y = Mathf.Max(maxSize.y, pos.y);
            }
            var newSize = new Vector3Int(1 + maxSize.x - minSize.x, 1 + maxSize.y - minSize.y, oldSize.z);
            var newPivot = new Vector3Int(-minSize.x, -minSize.y, oldPivot.z);
            UpdateSizeAndPivot(newSize, new Vector3Int(newPivot.x, newPivot.y, newPivot.z));
            foreach (Vector3Int oldPos in oldBounds.allPositionsWithin)
            {
                if (oldCells[GetCellIndex(oldPos.x, oldPos.y, oldPos.z, oldSize.x, oldSize.y, oldSize.z)].tile == null)
                    continue;
                var newPos = RotateHexagonPosition(direction, new Vector3Int(oldPos.x, oldPos.y, oldPos.z) - oldPivot) + newPivot;
                m_Cells[GetCellIndex(newPos.x, newPos.y, newPos.z)] = oldCells[GetCellIndex(oldPos.x, oldPos.y, oldPos.z, oldSize.x, oldSize.y, oldSize.z)];
            }

            Matrix4x4 rotation = direction == RotationDirection.Clockwise ? s_HexagonClockwise : s_HexagonCounterClockwise;
            Matrix4x4 counterRotation = direction != RotationDirection.Clockwise ? s_HexagonClockwise : s_HexagonCounterClockwise;
            foreach (BrushCell cell in m_Cells)
            {
                Matrix4x4 oldMatrix = cell.matrix;
                bool counter = (oldMatrix.lossyScale.x < 0) ^ (oldMatrix.lossyScale.y < 0);
                cell.matrix = oldMatrix * (counter ? counterRotation : rotation);
            }
        }

        private static Vector3Int HexagonToCube(Vector3Int position)
        {
            var cube = Vector3Int.zero;
            cube.x = position.x - (position.y - (position.y & 1)) / 2;
            cube.z = position.y;
            cube.y = -cube.x - cube.z;
            return cube;
        }

        private static Vector3Int CubeToHexagon(Vector3Int position)
        {
            var hexagon = Vector3Int.zero;
            hexagon.x = position.x + (position.z - (position.z & 1)) / 2;
            hexagon.y = position.z;
            hexagon.z = 0;
            return hexagon;
        }

        /// <summary>Flips the brush in the given axis.</summary>
        /// <param name="flip">Axis to flip by.</param>
        /// <param name="layout">Cell Layout for flipping.</param>
        public override void Flip(FlipAxis flip, GridLayout.CellLayout layout)
        {
            if (flip == FlipAxis.X)
                FlipX(layout);
            else
                FlipY(layout);
        }

        /// <summary>Picks tiles from selected Tilemaps and child GameObjects, given the coordinates of the cells.</summary>
        /// <param name="gridLayout">Grid to pick data from.</param>
        /// <param name="brushTarget">Target of the picking operation. By default the currently selected GameObject.</param>
        /// <param name="position">The coordinates of the cells to paint data from.</param>
        /// <param name="pickStart">Pivot of the picking brush.</param>
        public override void Pick(GridLayout gridLayout, GameObject brushTarget, BoundsInt position, Vector3Int pickStart)
        {
            Reset();
            UpdateSizeAndPivot(new Vector3Int(position.size.x, position.size.y, 1), new Vector3Int(pickStart.x, pickStart.y, 0));

            m_LastPickedLayout = gridLayout.cellLayout;
            m_LastPickedCellSize = gridLayout.cellSize;
            m_LastPickedCellGap = gridLayout.cellGap;
            m_LastPickedCellSwizzle = gridLayout.cellSwizzle;

            if (brushTarget == null)
                return;

            var tilemap = brushTarget.GetComponent<Tilemap>();
            foreach (var pos in position.allPositionsWithin)
            {
                var brushPosition = new Vector3Int(pos.x - position.x, pos.y - position.y, 0);
                PickCell(pos, brushPosition, tilemap);
            }
        }

        private void PickCell(Vector3Int position, Vector3Int brushPosition, Tilemap tilemap)
        {
            if (tilemap == null)
                return;

            SetTile(brushPosition, tilemap.GetTile(position));
            SetMatrix(brushPosition, tilemap.GetTransformMatrix(position));
            SetColor(brushPosition, tilemap.GetColor(position));
        }

        private void StoreCells()
        {
            m_StoredSize = m_Size;
            m_StoredPivot = m_Pivot;
            if (m_Cells != null)
            {
                m_StoredCells = new BrushCell[m_Cells.Length];
                for (int i = 0; i < m_Cells.Length; ++i)
                {
                    m_StoredCells[i] = m_Cells[i];
                }
            }
            else
            {
                m_StoredCells = Array.Empty<BrushCell>();
            }
        }

        private void RestoreCells()
        {
            m_Size = m_StoredSize;
            m_Pivot = m_StoredPivot;
            if (m_StoredCells != null)
            {
                m_Cells = new BrushCell[m_StoredCells.Length];
                m_TileChangeDataList = new List<TileChangeData>(m_StoredCells.Length);
                for (int i = 0; i < m_StoredCells.Length; ++i)
                {
                    m_Cells[i] = m_StoredCells[i];
                }
            }
        }

        /// <summary>MoveStart is called when user starts moving the area previously selected with the selection marquee.</summary>
        /// <param name="gridLayout">Grid used for layout.</param>
        /// <param name="brushTarget">Target of the move operation. By default the currently selected GameObject.</param>
        /// <param name="position">Position where the move operation has started.</param>
        public override void MoveStart(GridLayout gridLayout, GameObject brushTarget, BoundsInt position)
        {
            Tilemap tilemap = brushTarget.GetComponent<Tilemap>();
            if (tilemap == null)
                return;

            StoreCells();
            Reset();
            UpdateSizeAndPivot(new Vector3Int(position.size.x, position.size.y, 1), Vector3Int.zero);

            foreach (Vector3Int pos in position.allPositionsWithin)
            {
                Vector3Int brushPosition = new Vector3Int(pos.x - position.x, pos.y - position.y, 0);
                PickCell(pos, brushPosition, tilemap);
                tilemap.SetTile(pos, null);
            }
        }

        /// <summary>MoveEnd is called when user has ended the move of the area previously selected with the selection marquee.</summary>
        /// <param name="gridLayout">Grid used for layout.</param>
        /// <param name="brushTarget">Target of the move operation. By default the currently selected GameObject.</param>
        /// <param name="position">Position where the move operation has ended.</param>
        public override void MoveEnd(GridLayout gridLayout, GameObject brushTarget, BoundsInt position)
        {
            Paint(gridLayout, brushTarget, position.min);
            Reset();
            RestoreCells();
        }

        /// <summary>Clears all data of the brush.</summary>
        public void Reset()
        {
            UpdateSizeAndPivot(Vector3Int.one, Vector3Int.zero);
        }

        private void FlipX(GridLayout.CellLayout layout)
        {
            var oldCells = m_Cells.Clone() as BrushCell[];
            var oldBounds = new BoundsInt(Vector3Int.zero, m_Size);

            foreach (var oldPos in oldBounds.allPositionsWithin)
            {
                var newX = m_Size.x - oldPos.x - 1;
                var toIndex = GetCellIndex(newX, oldPos.y, oldPos.z);
                var fromIndex = GetCellIndex(oldPos);
                m_Cells[toIndex] = oldCells[fromIndex];
            }

            var newPivotX = m_Size.x - pivot.x - 1;
            pivot = new Vector3Int(newPivotX, pivot.y, pivot.z);

            if (layout == GridLayout.CellLayout.Hexagon)
            {
                FlipCellsHexagon(ref m_Cells, new Vector3(-1f, 1f, 1f));
            }
            else
            {
                FlipCells(ref m_Cells, new Vector3(-1f, 1f, 1f));
            }
        }

        private void FlipY(GridLayout.CellLayout layout)
        {
            var oldCells = m_Cells.Clone() as BrushCell[];
            var oldBounds = new BoundsInt(Vector3Int.zero, m_Size);

            foreach (var oldPos in oldBounds.allPositionsWithin)
            {
                var newY = m_Size.y - oldPos.y - 1;
                var toIndex = GetCellIndex(oldPos.x, newY, oldPos.z);
                var fromIndex = GetCellIndex(oldPos);
                m_Cells[toIndex] = oldCells[fromIndex];
            }

            var newPivotY = m_Size.y - pivot.y - 1;
            pivot = new Vector3Int(pivot.x, newPivotY, pivot.z);
            if (layout == GridLayout.CellLayout.Hexagon)
            {
                FlipCellsHexagon(ref m_Cells, new Vector3(1f, -1f, 1f));
            }
            else
            {
                FlipCells(ref m_Cells, new Vector3(1f, -1f, 1f));
            }
        }

        private static void FlipCellsHexagon(ref BrushCell[] cells, Vector3 scale)
        {
            foreach (var cell in cells)
            {
                var oldMatrix = cell.matrix;
                var oldScale = cell.matrix.lossyScale;
                var unflipScale = Vector3.one;
                if (oldScale.x < 0)
                {
                    unflipScale.x = -1f;
                }
                if (oldScale.y < 0)
                {
                    unflipScale.y = -1f;
                }
                var unflip = Matrix4x4.TRS(Vector3.zero, Quaternion.identity, unflipScale);
                var unflipMatrix = oldMatrix * unflip;
                var angles = unflipMatrix.rotation.eulerAngles;

                if (Mathf.Approximately(angles.x, 0f) && Mathf.Approximately(angles.y, 0f))
                {
                    var reversedAngles = 360 - angles.z;
                    var newScale = Vector3.Scale(oldScale, scale);
                    var newMatrix = Matrix4x4.TRS(oldMatrix.GetPosition(), Quaternion.Euler(0f, 0f, reversedAngles), newScale);
                    cell.matrix = newMatrix;
                }
                else
                {
                    var flip = Matrix4x4.TRS(Vector3.zero, Quaternion.identity, scale);
                    cell.matrix = oldMatrix * flip;
                }
            }
        }

        private static void FlipCells(ref BrushCell[] cells, Vector3 scale)
        {
            var flip = Matrix4x4.TRS(Vector3.zero, Quaternion.identity, scale);
            foreach (BrushCell cell in cells)
            {
                Matrix4x4 oldMatrix = cell.matrix;
                if (Mathf.Approximately(oldMatrix.rotation.x + oldMatrix.rotation.y + oldMatrix.rotation.z + oldMatrix.rotation.w, 1.0f))
                    cell.matrix = oldMatrix * flip;
                else
                    cell.matrix = oldMatrix * s_180Rotate * flip;
            }
        }

        /// <summary>Updates the size, pivot and the number of layers of the brush.</summary>
        /// <param name="newSize">New size of the brush.</param>
        /// <param name="newPivot">New pivot of the brush.</param>
        public void UpdateSizeAndPivot(Vector3Int newSize, Vector3Int newPivot)
        {
            m_Size = newSize;
            m_Pivot = newPivot;
            SizeUpdated();
        }

        /// <summary>Sets a Tile at the position in the brush.</summary>
        /// <param name="position">Position to set the tile in the brush.</param>
        /// <param name="tile">Tile to set in the brush.</param>
        public void SetTile(Vector3Int position, TileBase tile)
        {
            if (ValidateCellPosition(position))
                m_Cells[GetCellIndex(position)].tile = tile;
        }

        /// <summary>Sets a transform matrix at the position in the brush. This matrix is used specifically for tiles on a Tilemap and not GameObjects of the brush cell.</summary>
        /// <param name="position">Position to set the transform matrix in the brush.</param>
        /// <param name="matrix">Transform matrix to set in the brush.</param>
        public void SetMatrix(Vector3Int position, Matrix4x4 matrix)
        {
            if (ValidateCellPosition(position))
                m_Cells[GetCellIndex(position)].matrix = matrix;
        }

        /// <summary>Sets a tint color at the position in the brush.</summary>
        /// <param name="position">Position to set the color in the brush.</param>
        /// <param name="color">Tint color to set in the brush.</param>
        public void SetColor(Vector3Int position, Color color)
        {
            if (ValidateCellPosition(position))
                m_Cells[GetCellIndex(position)].color = color;
        }

        /// <summary>Gets the index to the GridBrush::ref::BrushCell based on the position of the BrushCell.</summary>
        /// <param name="brushPosition">Position of the BrushCell.</param>
        /// <returns>The index to the GridBrush::ref::BrushCell.</returns>
        public int GetCellIndex(Vector3Int brushPosition)
        {
            return GetCellIndex(brushPosition.x, brushPosition.y, brushPosition.z);
        }

        /// <summary>Gets the index to the GridBrush::ref::BrushCell based on the position of the BrushCell.</summary>
        /// <param name="x">X Position of the BrushCell.</param>
        /// <param name="y">Y Position of the BrushCell.</param>
        /// <param name="z">Z Position of the BrushCell.</param>
        /// <returns>The index to the GridBrush::ref::BrushCell.</returns>
        public int GetCellIndex(int x, int y, int z)
        {
            return x + m_Size.x * y + m_Size.x * m_Size.y * z;
        }

        /// <summary>Gets the index to the GridBrush::ref::BrushCell based on the position of the BrushCell.</summary>
        /// <param name="x">X Position of the BrushCell.</param>
        /// <param name="y">Y Position of the BrushCell.</param>
        /// <param name="z">Z Position of the BrushCell.</param>
        /// <param name="sizex">X Size of Brush.</param>
        /// <param name="sizey">Y Size of Brush.</param>
        /// <param name="sizez">Z Size of Brush.</param>
        /// <returns>The index to the GridBrush::ref::BrushCell.</returns>
        public int GetCellIndex(int x, int y, int z, int sizex, int sizey, int sizez)
        {
            return x + sizex * y + sizex * sizey * z;
        }

        /// <summary>Gets the index to the GridBrush::ref::BrushCell based on the position of the BrushCell. Wraps each coordinate if it is larger than the size of the GridBrush.</summary>
        /// <param name="x">X Position of the BrushCell.</param>
        /// <param name="y">Y Position of the BrushCell.</param>
        /// <param name="z">Z Position of the BrushCell.</param>
        /// <returns>The index to the GridBrush::ref::BrushCell.</returns>
        public int GetCellIndexWrapAround(int x, int y, int z)
        {
            return (x % m_Size.x) + m_Size.x * (y % m_Size.y) + m_Size.x * m_Size.y * (z % m_Size.z);
        }

        private bool ValidateCellPosition(Vector3Int position)
        {
            var valid =
                position.x >= 0 && position.x < size.x &&
                position.y >= 0 && position.y < size.y &&
                position.z >= 0 && position.z < size.z;
            if (!valid)
                throw new ArgumentException(
                    $"Position {position} is an invalid cell position. Valid range is between [{Vector3Int.zero}, {size}).");
            return true;
        }

        private void SizeUpdated()
        {
            var cellSize = m_Size.x * m_Size.y * m_Size.z;
            m_Cells = new BrushCell[cellSize];
            m_TileChangeDataList = new List<TileChangeData>(cellSize);
            var bounds = new BoundsInt(Vector3Int.zero, m_Size);
            foreach (var pos in bounds.allPositionsWithin)
            {
                m_Cells[GetCellIndex(pos)] = new BrushCell();
            }
        }

        /// <summary>
        /// Returns a HashCode for the GridBrush based on its contents.
        /// </summary>
        /// <returns>A HashCode for the GridBrush based on its contents.</returns>
        public override int GetHashCode()
        {
            int hash = 0;
            unchecked
            {
                hash = hash * 33 + size.GetHashCode();
                hash = hash * 33 + pivot.GetHashCode();
                foreach (var cell in cells)
                {
                    hash = hash * 33 + cell.GetHashCode();
                }
            }
            return hash;
        }

        /// <summary>Brush Cell stores the data to be painted in a grid cell.</summary>
        [Serializable]
        public class BrushCell
        {
            /// <summary>Tile to be placed when painting.</summary>
            public TileBase tile { get { return m_Tile; } set { m_Tile = value; } }
            /// <summary>The transform matrix of the brush cell.</summary>
            public Matrix4x4 matrix { get { return m_Matrix; } set { m_Matrix = value; } }
            /// <summary>Color to tint the tile when painting.</summary>
            public Color color { get { return m_Color; } set { m_Color = value; } }

            [SerializeField] private TileBase m_Tile;
            [SerializeField] Matrix4x4 m_Matrix = Matrix4x4.identity;
            [SerializeField] private Color m_Color = Color.white;

            /// <summary>
            /// Returns a HashCode for the BrushCell based on its contents.
            /// </summary>
            /// <returns>A HashCode for the BrushCell based on its contents.</returns>
            public override int GetHashCode()
            {
                int hash;
                unchecked
                {
                    hash = tile != null ? tile.GetInstanceID() : 0;
                    hash = hash * 33 + matrix.GetHashCode();
                    hash = hash * 33 + matrix.rotation.GetHashCode();
                    hash = hash * 33 + color.GetHashCode();
                }
                return hash;
            }
        }
    }
}
