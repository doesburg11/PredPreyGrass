using UnityEngine;

namespace UnityEditor.U2D.Animation
{
    internal class MeshTool : BaseTool
    {
        MeshCache m_Mesh;
        ISelection<int> m_SelectionOverride;
        SpriteMeshController m_SpriteMeshController;
        SpriteMeshView m_SpriteMeshView;
        RectSelectionTool<int> m_RectSelectionTool = new RectSelectionTool<int>();
        RectVertexSelector m_RectVertexSelector = new RectVertexSelector();
        UnselectTool<int> m_UnselectTool = new UnselectTool<int>();
        ITriangulator m_Triangulator;

        public MeshCache mesh => m_Mesh;

        public SpriteMeshViewMode mode
        {
            set => m_SpriteMeshView.mode = value;
        }

        public bool disable
        {
            set => m_SpriteMeshController.disable = value;
        }

        public ISelection<int> selectionOverride
        {
            get => m_SelectionOverride;
            set => m_SelectionOverride = value;
        }

        public override int defaultControlID
        {
            get
            {
                if (m_Mesh == null)
                    return 0;

                return m_RectSelectionTool.controlID;
            }
        }

        ISelection<int> selection
        {
            get
            {
                if (selectionOverride != null)
                    return selectionOverride;
                return skinningCache.vertexSelection;
            }
        }

        internal override void OnCreate()
        {
            m_SpriteMeshController = new SpriteMeshController();
            m_SpriteMeshView = new SpriteMeshView(new GUIWrapper());
            m_Triangulator = new Triangulator();
        }

        protected override void OnActivate()
        {
            m_SpriteMeshController.disable = false;
            m_SelectionOverride = null;

            SetupSprite(skinningCache.selectedSprite);

            skinningCache.events.selectedSpriteChanged.AddListener(OnSelectedSpriteChanged);
        }

        protected override void OnDeactivate()
        {
            skinningCache.events.selectedSpriteChanged.RemoveListener(OnSelectedSpriteChanged);
        }

        void OnSelectedSpriteChanged(SpriteCache sprite)
        {
            SetupSprite(sprite);
        }

        internal void SetupSprite(SpriteCache sprite)
        {
            var mesh = sprite.GetMesh();

            if (m_Mesh != null
                && m_Mesh != mesh
                && selection.Count > 0)
                selection.Clear();

            m_Mesh = mesh;
            m_SpriteMeshController.spriteMeshData = m_Mesh;
        }

        void SetupGUI()
        {
            m_SpriteMeshController.spriteMeshView = m_SpriteMeshView;
            m_SpriteMeshController.triangulator = m_Triangulator;
            m_SpriteMeshController.cacheUndo = skinningCache;
            m_RectSelectionTool.cacheUndo = skinningCache;
            m_RectSelectionTool.rectSelector = m_RectVertexSelector;
            m_RectVertexSelector.selection = selection;
            m_UnselectTool.cacheUndo = skinningCache;
            m_UnselectTool.selection = selection;

            m_SpriteMeshController.frame = new Rect(Vector2.zero, m_Mesh.sprite.textureRect.size);
            m_SpriteMeshController.selection = selection;
            m_SpriteMeshView.defaultControlID = defaultControlID;
            m_RectVertexSelector.spriteMeshData = m_Mesh;
        }

        protected override void OnGUI()
        {
            if (m_Mesh == null)
                return;

            SetupGUI();

            var handlesMatrix = Handles.matrix;
            Handles.matrix *= m_Mesh.sprite.GetLocalToWorldMatrixFromMode();

            BeginPositionOverride();

            EditorGUI.BeginChangeCheck();

            var guiEnabled = GUI.enabled;
            var moveAction = m_SpriteMeshController.spriteMeshView.IsActionHot(MeshEditorAction.MoveEdge) || m_SpriteMeshController.spriteMeshView.IsActionHot(MeshEditorAction.MoveVertex);
            GUI.enabled = (!skinningCache.IsOnVisualElement() && guiEnabled) || moveAction;
            m_SpriteMeshController.OnGUI();
            GUI.enabled = guiEnabled;

            if (EditorGUI.EndChangeCheck())
                UpdateMesh();

            m_RectSelectionTool.OnGUI();
            m_UnselectTool.OnGUI();

            Handles.matrix = handlesMatrix;

            EndPositionOverride();
        }

        public void BeginPositionOverride()
        {
            if (m_Mesh != null)
            {
                var skeleton = skinningCache.GetEffectiveSkeleton(m_Mesh.sprite);
                Debug.Assert(skeleton != null);

                if (skeleton.isPosePreview)
                {
                    var overrideVertices = m_Mesh.sprite.GetMeshPreview().vertices;
                    var convertedVerts = new Vector2[overrideVertices.Count];
                    for (var i = 0; i < convertedVerts.Length; ++i)
                        convertedVerts[i] = new Vector2(overrideVertices[i].x, overrideVertices[i].y);
                    m_Mesh.SetVertexPositionsOverride(convertedVerts);
                }
            }
        }

        public void EndPositionOverride()
        {
            if (m_Mesh != null)
                m_Mesh.ClearVertexPositionOverride();
        }

        public void UpdateWeights()
        {
            InvokeMeshChanged();
        }

        public void UpdateMesh()
        {
            InvokeMeshChanged();
        }

        void InvokeMeshChanged()
        {
            if (m_Mesh != null)
                skinningCache.events.meshChanged.Invoke(m_Mesh);
        }
    }
}
