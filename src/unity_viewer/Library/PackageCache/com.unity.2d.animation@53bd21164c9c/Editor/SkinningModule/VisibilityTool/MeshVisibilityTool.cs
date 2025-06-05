using System.Collections.Generic;
using UnityEngine;
using UnityEditor.IMGUI.Controls;
using System;
using System.Linq;
using UnityEngine.UIElements;

namespace UnityEditor.U2D.Animation
{
    internal class MeshVisibilityTool : IVisibilityTool
    {
        MeshVisibilityToolView m_View;
        MeshVisibilityToolModel m_Model;
        public SkinningCache skinningCache { get; }

        public MeshVisibilityTool(SkinningCache s)
        {
            skinningCache = s;
        }

        public void Setup()
        {
            m_Model = skinningCache.CreateCache<MeshVisibilityToolModel>();
            m_View = new MeshVisibilityToolView(skinningCache)
            {
                getModel = () => m_Model,
                setAllVisibility = SetAllVisibility,
                getAllVisibility = GetAllVisibility
            };
        }

        public void Dispose() { }

        public VisualElement view => m_View;
        public string name => "Mesh";

        public void Activate()
        {
            skinningCache.events.selectedSpriteChanged.AddListener(OnSelectionChange);
            skinningCache.events.skinningModeChanged.AddListener(OnViewModeChanged);
            OnViewModeChanged(skinningCache.mode);
            if (m_Model.previousVisibility != m_Model.allVisibility)
            {
                SetAllMeshVisibility();
                m_Model.previousVisibility = m_Model.allVisibility;
            }
        }

        public void Deactivate()
        {
            skinningCache.events.selectedSpriteChanged.RemoveListener(OnSelectionChange);
            skinningCache.events.skinningModeChanged.RemoveListener(OnViewModeChanged);
        }

        public bool isAvailable => false;

        public void SetAvailabilityChangeCallback(Action callback) {}

        void OnViewModeChanged(SkinningMode characterMode)
        {
            if (characterMode == SkinningMode.Character)
            {
                m_View.Setup(skinningCache.GetSprites());
            }
            else
            {
                m_View.Setup(new[] { skinningCache.selectedSprite });
            }
        }

        void OnSelectionChange(SpriteCache sprite)
        {
            OnViewModeChanged(skinningCache.mode);
            SetAllMeshVisibility();
        }

        void SetAllVisibility(bool visibility)
        {
            using (skinningCache.UndoScope(TextContent.meshVisibility))
            {
                m_Model.allVisibility = visibility;
                SetAllMeshVisibility();
            }
        }

        void SetAllMeshVisibility()
        {
            SpriteCache[] sprites;
            if (skinningCache.mode == SkinningMode.Character)
                sprites = skinningCache.GetSprites();
            else
                sprites = new[] { skinningCache.selectedSprite };

            foreach (var spr in sprites)
            {
                if (spr != null)
                    MeshVisibilityToolModel.SetMeshVisibility(spr, m_Model.allVisibility);
            }
        }

        bool GetAllVisibility()
        {
            return m_Model.allVisibility;
        }
    }

    internal class MeshVisibilityToolModel : SkinningObject
    {
        [SerializeField]
        bool m_AllVisibility = true;

        public bool allVisibility
        {
            get => m_AllVisibility;
            set => m_AllVisibility = value;
        }

        public static void SetMeshVisibility(SpriteCache sprite, bool visibility) { }

        public static bool GetMeshVisibility(SpriteCache sprite)
        {
            return false;
        }

        public bool ShouldDisable(SpriteCache sprite)
        {
            var mesh = sprite.GetMesh();
            return mesh == null || mesh.vertices.Length == 0;
        }

        public bool previousVisibility { get; set; } = true;
    }

    internal class MeshVisibilityToolView : VisibilityToolViewBase
    {
        public Func<MeshVisibilityToolModel> getModel = () => null;
        public Action<bool> setAllVisibility = (b) => {};
        public Func<bool> getAllVisibility = () => true;
        public SkinningCache skinningCache { get; set; }

        public MeshVisibilityToolView(SkinningCache s)
        {
            skinningCache = s;
            var columns = new MultiColumnHeaderState.Column[2];
            columns[0] = new MultiColumnHeaderState.Column
            {
                headerContent = new GUIContent(TextContent.name),
                headerTextAlignment = TextAlignment.Center,
                width = 200,
                minWidth = 130,
                autoResize = true,
                allowToggleVisibility = false
            };
            columns[1] = new MultiColumnHeaderState.Column
            {
                headerContent = new GUIContent(EditorGUIUtility.FindTexture("visibilityOn")),
                headerTextAlignment = TextAlignment.Center,
                width = 32,
                minWidth = 32,
                maxWidth = 32,
                autoResize = false,
                allowToggleVisibility = true
            };
            var multiColumnHeaderState = new MultiColumnHeaderState(columns);
            var multiColumnHeader = new VisibilityToolColumnHeader(multiColumnHeaderState)
            {
                GetAllVisibility = InternalGetAllVisibility,
                SetAllVisibility = InternalSetAllVisibility,
                canSort = false,
                height = 20,
                visibilityColumn = 1
            };
            m_TreeView = new MeshTreeView(m_TreeViewState, multiColumnHeader)
            {
                GetModel = InternalGetModel
            };
            SetupSearchField();
        }

        MeshVisibilityToolModel InternalGetModel()
        {
            return getModel();
        }

        public void Setup(SpriteCache[] sprites)
        {
            ((MeshTreeView)m_TreeView).Setup(sprites);
            ((MeshTreeView)m_TreeView).SetSelection(skinningCache.selectedSprite);
        }

        bool InternalGetAllVisibility()
        {
            return getAllVisibility();
        }

        void InternalSetAllVisibility(bool visibility)
        {
            setAllVisibility(visibility);
        }
    }

    class MeshTreeView : VisibilityTreeViewBase
    {
        private List<SpriteCache> m_Sprites = new List<SpriteCache>();

        public MeshTreeView(TreeViewState treeViewState, MultiColumnHeader header)
            : base(treeViewState, header)
        {
            this.showAlternatingRowBackgrounds = true;
            this.useScrollView = true;
            Reload();
        }

        public Func<MeshVisibilityToolModel> GetModel = () => null;

        public void Setup(SpriteCache[] sprites)
        {
            m_Sprites.Clear();
            m_Sprites.AddRange(sprites);
            Reload();
        }

        private static TreeViewItem CreateTreeViewItem(SpriteCache part)
        {
            return new TreeViewItemBase<SpriteCache>(part.GetInstanceID(), -1, part.name, part);
        }

        private void AddTreeViewItem(IList<TreeViewItem> rows, SpriteCache part)
        {
            if (string.IsNullOrEmpty(searchString) || part.name.IndexOf(searchString, StringComparison.OrdinalIgnoreCase) >= 0)
            {
                var item = CreateTreeViewItem(part);
                rows.Add(item);
                rootItem.AddChild(item);
            }
        }

        private void CellGUI(Rect cellRect, TreeViewItem item, int column, ref RowGUIArgs args)
        {
            CenterRectUsingSingleLineHeight(ref cellRect);
            switch (column)
            {
                case 0:
                    DrawNameCell(cellRect, item, ref args);
                    break;
                case 1:
                    DrawVisibilityCell(cellRect, item);
                    break;
            }
        }

        private void DrawVisibilityCell(Rect cellRect, TreeViewItem item)
        {
            GUIStyle style = MultiColumnHeader.DefaultStyles.columnHeaderCenterAligned;
            var itemView = item as TreeViewItemBase<SpriteCache>;
            var shouldDisable = GetModel().ShouldDisable(itemView.customData);
            using (new EditorGUI.DisabledScope(shouldDisable))
            {
                EditorGUI.BeginChangeCheck();
                bool visible = MeshVisibilityToolModel.GetMeshVisibility(itemView.customData);
                visible = GUI.Toggle(cellRect, visible, visible ? VisibilityIconStyle.visibilityOnIcon : VisibilityIconStyle.visibilityOffIcon, style);
                if (EditorGUI.EndChangeCheck())
                    MeshVisibilityToolModel.SetMeshVisibility(itemView.customData, visible);
            }
        }

        private void DrawNameCell(Rect cellRect, TreeViewItem item, ref RowGUIArgs args)
        {
            args.rowRect = cellRect;
            base.RowGUI(args);
        }

        protected override void RowGUI(RowGUIArgs args)
        {
            var item = args.item;

            for (int i = 0; i < args.GetNumVisibleColumns(); ++i)
            {
                CellGUI(args.GetCellRect(i), item, args.GetColumn(i), ref args);
            }
        }

        protected override void SelectionChanged(IList<int> selectedIds)
        {
            SpriteCache newSelected = null;
            if (selectedIds.Count > 0)
            {
                var selected = GetRows().FirstOrDefault(x => ((TreeViewItemBase<SpriteCache>)x).customData.GetInstanceID() == selectedIds[0]) as TreeViewItemBase<SpriteCache>;
                if (selected != null)
                    newSelected = selected.customData;
            }

            var skinningCache = newSelected.skinningCache;

            using (skinningCache.UndoScope(TextContent.selectionChange))
            {
                skinningCache.events.selectedSpriteChanged.Invoke(newSelected);
            }
        }

        public void SetSelection(SpriteCache sprite)
        {
            var rows = GetRows();
            for (int i = 0; rows != null && i < rows.Count; ++i)
            {
                var r = (TreeViewItemBase<SpriteCache>)rows[i];
                if (r.customData == sprite)
                {
                    SetSelection(new[] { r.customData.GetInstanceID() }, TreeViewSelectionOptions.RevealAndFrame);
                    break;
                }
            }
        }

        protected override IList<TreeViewItem> BuildRows(TreeViewItem root)
        {
            var rows = GetRows() ?? new List<TreeViewItem>(200);
            rows.Clear();

            m_Sprites.RemoveAll(s => s == null);

            foreach (var sprite in m_Sprites)
                AddTreeViewItem(rows, sprite);

            SetupDepthsFromParentsAndChildren(root);
            return rows;
        }
    }
}
