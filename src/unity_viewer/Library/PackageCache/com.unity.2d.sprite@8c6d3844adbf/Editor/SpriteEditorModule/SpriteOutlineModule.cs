using UnityEngine;
using System.Collections.Generic;
using System;
using System.Linq;
using Unity.Collections;
using UnityEditor.U2D.Sprites.SpriteEditorTool;
using UnityEngine.U2D;
using UnityEngine.UIElements;
using Object = UnityEngine.Object;

namespace UnityEditor.U2D.Sprites
{
    // We need this so that undo/redo works
    [Serializable]
    internal class SpriteOutline
    {
        [SerializeField]
        public List<Vector2> m_Path = new List<Vector2>();

        public void Add(Vector2 point)
        {
            m_Path.Add(point);
        }

        public void Insert(int index, Vector2 point)
        {
            m_Path.Insert(index, point);
        }

        public void RemoveAt(int index)
        {
            m_Path.RemoveAt(index);
        }

        public Vector2 this[int index]
        {
            get { return m_Path[index]; }
            set { m_Path[index] = value; }
        }

        public int Count
        {
            get { return m_Path.Count; }
        }

        public void AddRange(IEnumerable<Vector2> addRange)
        {
            m_Path.AddRange(addRange);
        }
    }

    // Collection of outlines for a single Sprite
    [Serializable]
    internal class SpriteOutlineList
    {
        [SerializeField]
        List<SpriteOutline> m_SpriteOutlines;
        [SerializeField]
        float m_TessellationDetail = 0;

        public List<SpriteOutline> spriteOutlines { get { return m_SpriteOutlines; } set { m_SpriteOutlines = value; } }
        public GUID spriteID { get; private set; }

        public float tessellationDetail
        {
            get { return m_TessellationDetail; }
            set
            {
                m_TessellationDetail = value;
                m_TessellationDetail = Mathf.Min(1, m_TessellationDetail);
                m_TessellationDetail = Mathf.Max(0, m_TessellationDetail);
            }
        }

        public SpriteOutlineList(GUID guid)
        {
            this.spriteID = guid;
            m_SpriteOutlines = new List<SpriteOutline>();
        }

        public SpriteOutlineList(GUID guid, List<Vector2[]> list)
        {
            this.spriteID = guid;

            m_SpriteOutlines = new List<SpriteOutline>(list.Count);
            for (int i = 0; i < list.Count; ++i)
            {
                var newList = new SpriteOutline();
                newList.m_Path.AddRange(list[i]);
                m_SpriteOutlines.Add(newList);
            }
        }

        public SpriteOutlineList(GUID guid, List<SpriteOutline> list)
        {
            this.spriteID = guid;

            m_SpriteOutlines = list;
        }

        public List<Vector2[]> ToListVector()
        {
            var value = new List<Vector2[]>(m_SpriteOutlines.Count);
            foreach (var s in m_SpriteOutlines)
            {
                value.Add(s.m_Path.ToArray());
            }
            return value;
        }

        public List<Vector2[]> ToListVectorCapped(Rect rect)
        {
            var value = ToListVector();
            rect.center = Vector2.zero;
            foreach (var path in value)
            {
                for (int i = 0; i < path.Length; ++i)
                {
                    var point = path[i];
                    path[i] = SpriteOutlineModule.CapPointToRect(point, rect);
                }
            }
            return value;
        }

        public SpriteOutline this[int index]
        {
            get { return IsValidIndex(index) ? m_SpriteOutlines[index] : null; }
            set
            {
                if (IsValidIndex(index))
                    m_SpriteOutlines[index] = value;
            }
        }

        public static implicit operator List<SpriteOutline>(SpriteOutlineList list)
        {
            return list != null ? list.m_SpriteOutlines : null;
        }

        public int Count { get { return m_SpriteOutlines.Count; } }

        bool IsValidIndex(int index)
        {
            return index >= 0 && index < Count;
        }
    }

    // Collection of Sprites' outlines
    internal class SpriteOutlineModel : ScriptableObject
    {
        [SerializeField]
        List<SpriteOutlineList> m_SpriteOutlineList = new List<SpriteOutlineList>();

        private SpriteOutlineModel()
        {}

        public SpriteOutlineList this[int index]
        {
            get { return IsValidIndex(index) ? m_SpriteOutlineList[index] : null; }
            set
            {
                if (IsValidIndex(index))
                    m_SpriteOutlineList[index] = value;
            }
        }

        public SpriteOutlineList this[GUID guid]
        {
            get { return m_SpriteOutlineList.FirstOrDefault(x => x.spriteID == guid); }
            set
            {
                var index = m_SpriteOutlineList.FindIndex(x => x.spriteID == guid);
                if (index != -1)
                    m_SpriteOutlineList[index] = value;
            }
        }

        public void AddListVector2(GUID guid, List<Vector2[]> outline)
        {
            m_SpriteOutlineList.Add(new SpriteOutlineList(guid, outline));
        }

        public int Count { get { return m_SpriteOutlineList.Count; } }

        bool IsValidIndex(int index)
        {
            return index >= 0 && index < Count;
        }
    }

    [RequireSpriteDataProvider(typeof(ISpriteOutlineDataProvider), typeof(ITextureDataProvider))]
    internal class SpriteOutlineModule : SpriteEditorModuleBase
    {
        class Styles
        {
            public GUIContent generatingOutlineDialogTitle = EditorGUIUtility.TrTextContent("Outline");
            public GUIContent generatingOutlineDialogContent = EditorGUIUtility.TrTextContent("Generating outline {0}/{1}");
            public Color spriteBorderColor = new Color(0.25f, 0.5f, 1f, 0.75f);
        }

        protected SpriteRect m_Selected;

        private const float k_HandleSize = 5f;
        private readonly string k_DeleteCommandName = EventCommandNames.Delete;
        private readonly string k_SoftDeleteCommandName = EventCommandNames.SoftDelete;

        private ShapeEditor[] m_ShapeEditors;
        private bool m_RequestRepaint;
        private Matrix4x4 m_HandleMatrix;
        private Vector2 m_MousePosition;
        private ShapeEditorRectSelectionTool m_ShapeSelectionUI;
        private bool m_WasRectSelecting = false;
        private Rect? m_SelectionRect;
        private ITexture2D m_OutlineTexture;
        private Styles m_Styles;
        protected SpriteOutlineModel m_Outline;
        protected ITextureDataProvider m_TextureDataProvider;
        protected SpriteOutlineToolOverlayPanel m_SpriteOutlineToolElement;

        [SerializeReference]
        private static SpriteOutlineList s_CopyOutline = null;

        public SpriteOutlineModule(ISpriteEditor sem, IEventSystem es, IUndoSystem us, IAssetDatabase ad, IGUIUtility gu, IShapeEditorFactory sef, ITexture2D outlineTexture)
        {
            spriteEditorWindow = sem;
            undoSystem = us;
            eventSystem = es;
            assetDatabase = ad;
            guiUtility = gu;
            shapeEditorFactory = sef;
            m_OutlineTexture = outlineTexture;

            m_ShapeSelectionUI = new ShapeEditorRectSelectionTool(gu);

            m_ShapeSelectionUI.RectSelect += RectSelect;
            m_ShapeSelectionUI.ClearSelection += ClearSelection;
        }

        public override string moduleName
        {
            get { return "Custom Outline"; }
        }

        public override bool ApplyRevert(bool apply)
        {
            if (m_Outline != null)
            {
                if (apply)
                {
                    var outlineDataProvider = spriteEditorWindow.GetDataProvider<ISpriteOutlineDataProvider>();
                    for (int i = 0; i < m_Outline.Count; ++i)
                    {
                        outlineDataProvider.SetOutlines(m_Outline[i].spriteID, m_Outline[i].ToListVector());
                        outlineDataProvider.SetTessellationDetail(m_Outline[i].spriteID, m_Outline[i].tessellationDetail);
                    }
                }

                Object.DestroyImmediate(m_Outline);
                m_Outline = null;
            }

            return true;
        }

        private Styles styles
        {
            get
            {
                if (m_Styles == null)
                    m_Styles = new Styles();
                return m_Styles;
            }
        }

        protected virtual List<SpriteOutline> selectedShapeOutline
        {
            get
            {
                return m_Outline[m_Selected.spriteID].spriteOutlines;
            }
            set
            {
                m_Outline[m_Selected.spriteID].spriteOutlines = value;
            }
        }

        protected virtual string alterateLabelText => L10n.Tr("From Physics Shape");

        private bool shapeEditorDirty
        {
            get; set;
        }

        private bool editingDisabled
        {
            get { return spriteEditorWindow.editingDisabled; }
        }

        private ISpriteEditor spriteEditorWindow
        {
            get; set;
        }

        private IUndoSystem undoSystem
        {
            get; set;
        }

        private IEventSystem eventSystem
        {
            get; set;
        }

        private IAssetDatabase assetDatabase
        {
            get; set;
        }

        private IGUIUtility guiUtility
        {
            get; set;
        }

        private IShapeEditorFactory shapeEditorFactory
        {
            get; set;
        }

        private void RectSelect(Rect r, ShapeEditor.SelectionType selectionType)
        {
            var localRect = EditorGUIExt.FromToRect(ScreenToLocal(r.min), ScreenToLocal(r.max));
            m_SelectionRect = localRect;
        }

        private void ClearSelection()
        {
            m_RequestRepaint = true;
        }

        protected virtual void LoadOutline()
        {
            m_Outline = ScriptableObject.CreateInstance<SpriteOutlineModel>();
            m_Outline.hideFlags = HideFlags.HideAndDontSave;
            var spriteDataProvider = spriteEditorWindow.GetDataProvider<ISpriteEditorDataProvider>();
            var outlineDataProvider = spriteEditorWindow.GetDataProvider<ISpriteOutlineDataProvider>();
            foreach (var rect in spriteDataProvider.GetSpriteRects())
            {
                var outlines = outlineDataProvider.GetOutlines(rect.spriteID);
                m_Outline.AddListVector2(rect.spriteID, outlines);
                m_Outline[m_Outline.Count - 1].tessellationDetail = outlineDataProvider.GetTessellationDetail(rect.spriteID);
            }
        }

        protected virtual List<Vector2[]> GetAlternateOutlines(GUID spriteID)
        {
            var alternateOutlineProvider = spriteEditorWindow.GetDataProvider<ISpritePhysicsOutlineDataProvider>();
            return alternateOutlineProvider.GetOutlines(spriteID);
        }

        public override void OnModuleActivate()
        {
            m_TextureDataProvider = spriteEditorWindow.GetDataProvider<ITextureDataProvider>();
            LoadOutline();
            GenerateOutlineIfNotExist();
            undoSystem.RegisterUndoCallback(UndoRedoPerformed);
            shapeEditorDirty = true;
            SetupShapeEditor();
            spriteEditorWindow.enableMouseMoveEvent = true;
            AddMainUI(spriteEditorWindow.GetMainVisualContainer());
        }

        void GenerateOutlineIfNotExist()
        {
            var rectCache = spriteEditorWindow.GetDataProvider<ISpriteEditorDataProvider>().GetSpriteRects();
            if (rectCache != null)
            {
                bool needApply = false;
                for (int i = 0; i < rectCache.Length; ++i)
                {
                    var rect = rectCache[i];
                    if (!HasShapeOutline(rect))
                    {
                        EditorUtility.DisplayProgressBar(styles.generatingOutlineDialogTitle.text,
                            string.Format(styles.generatingOutlineDialogContent.text, i + 1 , rectCache.Length),
                            (float)(i) / rectCache.Length);

                        SetupShapeEditorOutline(rect);
                        needApply = true;
                    }
                }
                if (needApply)
                {
                    EditorUtility.ClearProgressBar();
                    spriteEditorWindow.ApplyOrRevertModification(true);
                    LoadOutline();
                }
            }
        }

        public override void OnModuleDeactivate()
        {
            undoSystem.UnregisterUndoCallback(UndoRedoPerformed);
            CleanupShapeEditors();
            m_Selected = null;
            spriteEditorWindow.enableMouseMoveEvent = false;
            if (m_Outline != null)
            {
                undoSystem.ClearUndo(m_Outline);
                Object.DestroyImmediate(m_Outline);
                m_Outline = null;
            }
            RemoveMainUI(spriteEditorWindow.GetMainVisualContainer());
        }

        public override void DoMainGUI()
        {
            IEvent evt = eventSystem.current;

            m_RequestRepaint = false;
            m_HandleMatrix = Handles.matrix;

            m_MousePosition = Handles.inverseMatrix.MultiplyPoint(eventSystem.current.mousePosition);
            if (m_Selected == null || !m_Selected.rect.Contains(m_MousePosition) && !IsMouseOverOutlinePoints() && evt.shift == false)
                spriteEditorWindow.HandleSpriteSelection();

            HandleCreateNewOutline();

            m_WasRectSelecting = m_ShapeSelectionUI.isSelecting;

            UpdateShapeEditors();

            m_ShapeSelectionUI.OnGUI();

            DrawGizmos();

            if (m_RequestRepaint || evt.type == EventType.MouseMove)
                spriteEditorWindow.RequestRepaint();
        }

        protected virtual int alphaTolerance
        {
            get => SpriteOutlineModulePreference.alphaTolerance;
            set => SpriteOutlineModulePreference.alphaTolerance = value;
        }

        internal SpriteOutlineList selectedOutline => m_Outline[m_Selected.spriteID];

        public override void DoToolbarGUI(Rect drawArea)
        {}

        public override void DoPostGUI()
        {}

        public override bool CanBeActivated()
        {
            return SpriteFrameModule.GetSpriteImportMode(spriteEditorWindow.GetDataProvider<ISpriteEditorDataProvider>()) != SpriteImportMode.None;
        }

        private void RecordUndo()
        {
            undoSystem.RegisterCompleteObjectUndo(m_Outline, "Outline changed");
        }

        public void CreateNewOutline(Rect rectOutline)
        {
            Rect rect = m_Selected.rect;
            if (rect.Contains(rectOutline.min) && rect.Contains(rectOutline.max))
            {
                RecordUndo();
                SpriteOutline so = new SpriteOutline();
                Vector2 outlineOffset = new Vector2(0.5f * rect.width + rect.x, 0.5f * rect.height + rect.y);
                Rect selectionRect = new Rect(rectOutline);
                selectionRect.min = SnapPoint(rectOutline.min);
                selectionRect.max = SnapPoint(rectOutline.max);
                so.Add(CapPointToRect(new Vector2(selectionRect.xMin, selectionRect.yMin), rect) - outlineOffset);
                so.Add(CapPointToRect(new Vector2(selectionRect.xMax, selectionRect.yMin), rect) - outlineOffset);
                so.Add(CapPointToRect(new Vector2(selectionRect.xMax, selectionRect.yMax), rect) - outlineOffset);
                so.Add(CapPointToRect(new Vector2(selectionRect.xMin, selectionRect.yMax), rect) - outlineOffset);
                selectedShapeOutline.Add(so);
                spriteEditorWindow.SetDataModified();
                shapeEditorDirty = true;
            }
        }

        private void HandleCreateNewOutline()
        {
            if (m_WasRectSelecting && m_ShapeSelectionUI.isSelecting == false && m_SelectionRect != null && m_Selected != null)
            {
                bool createNewOutline = true;
                foreach (var se in m_ShapeEditors)
                {
                    if (se.selectedPoints.Count != 0)
                    {
                        createNewOutline = false;
                        break;
                    }
                }

                if (createNewOutline)
                    CreateNewOutline(m_SelectionRect.Value);
            }
            m_SelectionRect = null;
        }

        public void UpdateShapeEditors()
        {
            SetupShapeEditor();

            if (m_Selected != null)
            {
                IEvent currentEvent = eventSystem.current;
                var wantsDelete = currentEvent.type == EventType.ExecuteCommand && (currentEvent.commandName == k_SoftDeleteCommandName || currentEvent.commandName == k_DeleteCommandName);

                for (int i = 0; i < m_ShapeEditors.Length; ++i)
                {
                    if (m_ShapeEditors[i].GetPointsCount() == 0)
                        continue;

                    m_ShapeEditors[i].inEditMode = true;
                    m_ShapeEditors[i].OnGUI();
                    if (shapeEditorDirty)
                        break;
                }

                if (wantsDelete)
                {
                    // remove outline which have lesser than 3 points
                    for (int i = selectedShapeOutline.Count - 1; i >= 0; --i)
                    {
                        if (selectedShapeOutline[i].Count < 3)
                        {
                            selectedShapeOutline.RemoveAt(i);
                            shapeEditorDirty = true;
                        }
                    }
                }
            }
        }

        private bool IsMouseOverOutlinePoints()
        {
            if (m_Selected == null)
                return false;
            Vector2 outlineOffset = new Vector2(0.5f * m_Selected.rect.width + m_Selected.rect.x, 0.5f * m_Selected.rect.height + m_Selected.rect.y);
            float handleSize = GetHandleSize();
            Rect r = new Rect(0, 0, handleSize * 2, handleSize * 2);
            for (int i = 0; i < selectedShapeOutline.Count; ++i)
            {
                var outline = selectedShapeOutline[i];
                for (int j = 0; j < outline.Count; ++j)
                {
                    r.center = outline[j] + outlineOffset;
                    if (r.Contains(m_MousePosition))
                        return true;
                }
            }
            return false;
        }

        private float GetHandleSize()
        {
            return k_HandleSize / m_HandleMatrix.m00;
        }

        private void CleanupShapeEditors()
        {
            if (m_ShapeEditors != null)
            {
                for (int i = 0; i < m_ShapeEditors.Length; ++i)
                {
                    for (int j = 0; j < m_ShapeEditors.Length; ++j)
                    {
                        if (i != j)
                            m_ShapeEditors[j].UnregisterFromShapeEditor(m_ShapeEditors[i]);
                    }
                    m_ShapeEditors[i].OnDisable();
                }
            }
            m_ShapeEditors = null;
        }

        public void SetupShapeEditor()
        {
            if (shapeEditorDirty || m_Selected != spriteEditorWindow.selectedSpriteRect)
            {
                m_Selected = spriteEditorWindow.selectedSpriteRect;
                CleanupShapeEditors();

                if (m_Selected != null)
                {
                    if (!HasShapeOutline(m_Selected))
                        SetupShapeEditorOutline(m_Selected);
                    m_ShapeEditors = new ShapeEditor[selectedShapeOutline.Count];

                    for (int i = 0; i < selectedShapeOutline.Count; ++i)
                    {
                        int outlineIndex = i;
                        m_ShapeEditors[i] = shapeEditorFactory.CreateShapeEditor();
                        m_ShapeEditors[i].SetRectSelectionTool(m_ShapeSelectionUI);
                        m_ShapeEditors[i].LocalToWorldMatrix = () => m_HandleMatrix;
                        m_ShapeEditors[i].LocalToScreen = (point) => Handles.matrix.MultiplyPoint(point);
                        m_ShapeEditors[i].ScreenToLocal = ScreenToLocal;
                        m_ShapeEditors[i].RecordUndo = RecordUndo;
                        m_ShapeEditors[i].GetHandleSize = GetHandleSize;
                        m_ShapeEditors[i].lineTexture = m_OutlineTexture;
                        m_ShapeEditors[i].Snap = SnapPoint;
                        m_ShapeEditors[i].GetPointPosition = (index) => GetPointPosition(outlineIndex, index);
                        m_ShapeEditors[i].SetPointPosition = (index, position) => SetPointPosition(outlineIndex, index, position);
                        m_ShapeEditors[i].InsertPointAt = (index, position) => InsertPointAt(outlineIndex, index, position);
                        m_ShapeEditors[i].RemovePointAt = (index) => RemovePointAt(outlineIndex, index);
                        m_ShapeEditors[i].GetPointsCount = () => GetPointsCount(outlineIndex);
                    }
                    for (int i = 0; i < selectedShapeOutline.Count; ++i)
                    {
                        for (int j = 0; j < selectedShapeOutline.Count; ++j)
                        {
                            if (i != j)
                                m_ShapeEditors[j].RegisterToShapeEditor(m_ShapeEditors[i]);
                        }
                    }
                }
                else
                {
                    m_ShapeEditors = new ShapeEditor[0];
                }
            }
            shapeEditorDirty = false;
        }

        protected virtual bool HasShapeOutline(SpriteRect spriteRect)
        {
            var outline = m_Outline[spriteRect.spriteID] != null ? m_Outline[spriteRect.spriteID].spriteOutlines : null;
            return outline != null;
        }

        private void AddMainUI(VisualElement mainView)
        {
            m_SpriteOutlineToolElement = SpriteOutlineToolOverlayPanel.GenerateFromUXML(alterateLabelText);
            m_SpriteOutlineToolElement.AddStyleSheetPath("Packages/com.unity.2d.sprite/Editor/UI/SpriteEditor/SpriteEditor.uss");
            m_SpriteOutlineToolElement.AddToClassList("moduleWindow");
            m_SpriteOutlineToolElement.AddToClassList("bottomRightFloating");
            mainView.Add(m_SpriteOutlineToolElement);
            m_SpriteOutlineToolElement.onGenerateOutline += OnGenerateOutline;
            m_SpriteOutlineToolElement.onCopy += Copy;
            m_SpriteOutlineToolElement.onPaste += Paste;
            m_SpriteOutlineToolElement.onPasteAll += PasteAll;
            m_SpriteOutlineToolElement.onPasteAlternate += PasteFromAlternate;
            m_SpriteOutlineToolElement.onPasteAlternateAll += PasteAllFromAlternate;
            m_SpriteOutlineToolElement.onAlphaToleranceChanged += OnAlphaToleranceChanged;
            m_SpriteOutlineToolElement.onOutlineDetailChanged += OnOutlineDetailChanged;
            mainView.RegisterCallback<SpriteSelectionChangeEvent>(SpriteSelectionChange);
            SetupUIPanel();
        }

        private void RemoveMainUI(VisualElement mainView)
        {
            if (m_SpriteOutlineToolElement != null)
            {
                if (mainView.Contains(m_SpriteOutlineToolElement))
                    mainView.Remove(m_SpriteOutlineToolElement);
                mainView.UnregisterCallback<SpriteSelectionChangeEvent>(SpriteSelectionChange);
                m_SpriteOutlineToolElement.onGenerateOutline -= OnGenerateOutline;
                m_SpriteOutlineToolElement.onCopy -= Copy;
                m_SpriteOutlineToolElement.onPaste -= Paste;
                m_SpriteOutlineToolElement.onPasteAll -= PasteAll;
                m_SpriteOutlineToolElement.onPasteAlternate -= PasteFromAlternate;
                m_SpriteOutlineToolElement.onPasteAlternateAll -= PasteAllFromAlternate;
                m_SpriteOutlineToolElement.onAlphaToleranceChanged -= OnAlphaToleranceChanged;
                m_SpriteOutlineToolElement.onOutlineDetailChanged -= OnOutlineDetailChanged;
            }
        }

        void SetupUIPanel()
        {
            m_Selected = spriteEditorWindow.selectedSpriteRect;
            m_SpriteOutlineToolElement.SetPanelMode(m_Selected != null);
            if (m_Selected != null)
            {
                m_SpriteOutlineToolElement.outlineDetail = m_Outline[m_Selected.spriteID].tessellationDetail;
            }
            else
            {
                m_SpriteOutlineToolElement.outlineDetail = 0;
            }

            m_SpriteOutlineToolElement.alphaTolerance = alphaTolerance;
        }

        void OnAlphaToleranceChanged(int value)
        {
            alphaTolerance = value;
        }

        internal void OnOutlineDetailChanged(float value)
        {
            if(m_Selected != null)
                m_Outline[m_Selected.spriteID].tessellationDetail = value;
        }

        void OnGenerateOutline(bool forceGenerate)
        {
            RecordUndo();
            if (m_Selected != null)
            {
                selectedShapeOutline.Clear();
                SetupShapeEditorOutline(m_Selected);
            }
            else
            {
                var rectCache = spriteEditorWindow.GetDataProvider<ISpriteEditorDataProvider>().GetSpriteRects();
                if (rectCache != null)
                {
                    bool showedProgressBar = false;
                    for (int i = 0; i < rectCache.Length; ++i)
                    {
                        var rect = rectCache[i];
                        var outline = m_Outline[rect.spriteID] != null ? m_Outline[rect.spriteID].spriteOutlines : null;
                        if (forceGenerate || outline == null || outline.Count == 0)
                        {
                            EditorUtility.DisplayProgressBar(styles.generatingOutlineDialogTitle.text,
                                string.Format(styles.generatingOutlineDialogContent.text, i + 1, rectCache.Length),
                                (float)(i) / rectCache.Length);
                            showedProgressBar = true;
                            m_Outline[rect.spriteID].tessellationDetail = m_SpriteOutlineToolElement.outlineDetail;
                            SetupShapeEditorOutline(rect);
                        }
                    }
                    if(showedProgressBar)
                        EditorUtility.ClearProgressBar();
                }
            }
            spriteEditorWindow.SetDataModified();
            shapeEditorDirty = true;
        }

        void SpriteSelectionChange(SpriteSelectionChangeEvent evt)
        {
            var spriteRect = spriteEditorWindow.selectedSpriteRect;
            m_SpriteOutlineToolElement.SetPanelMode(spriteRect != null);
            if (spriteRect != null)
            {
                var data = m_Outline[spriteRect.spriteID];
                if(data != null)
                    m_SpriteOutlineToolElement.outlineDetail = m_Outline[spriteRect.spriteID].tessellationDetail;
            }
        }

        protected virtual void SetupShapeEditorOutline(SpriteRect spriteRect)
        {
            var outline = m_Outline[spriteRect.spriteID];
            var outlines = GenerateSpriteRectOutline(spriteRect.rect,
                Math.Abs(outline.tessellationDetail - (-1f)) < Mathf.Epsilon ? 0 : outline.tessellationDetail,
                (byte)(alphaTolerance), m_TextureDataProvider, m_SpriteOutlineToolElement.optimizeOutline);
            if (outlines.Count == 0)
            {
                Vector2 halfSize = spriteRect.rect.size * 0.5f;
                outlines = new List<SpriteOutline>()
                {
                    new SpriteOutline()
                    {
                        m_Path = new List<Vector2>()
                        {
                            new Vector2(-halfSize.x, -halfSize.y),
                            new Vector2(-halfSize.x, halfSize.y),
                            new Vector2(halfSize.x, halfSize.y),
                            new Vector2(halfSize.x, -halfSize.y),
                        }
                    }
                };
            }
            m_Outline[spriteRect.spriteID].spriteOutlines = outlines;
        }

        public Vector3 SnapPoint(Vector3 position)
        {
            if (m_SpriteOutlineToolElement.snapOn)
            {
                position.x = Mathf.RoundToInt(position.x);
                position.y = Mathf.RoundToInt(position.y);
            }
            return position;
        }

        public Vector3 GetPointPosition(int outlineIndex, int pointIndex)
        {
            if (outlineIndex >= 0 && outlineIndex < selectedShapeOutline.Count)
            {
                var outline = selectedShapeOutline[outlineIndex];
                if (pointIndex >= 0 && pointIndex < outline.Count)
                {
                    return ConvertSpriteRectSpaceToTextureSpace(outline[pointIndex]);
                }
            }
            return new Vector3(float.NaN, float.NaN, float.NaN);
        }

        public void SetPointPosition(int outlineIndex, int pointIndex, Vector3 position)
        {
            selectedShapeOutline[outlineIndex][pointIndex] = ConvertTextureSpaceToSpriteRectSpace(CapPointToRect(position, m_Selected.rect));
            spriteEditorWindow.SetDataModified();
        }

        public void InsertPointAt(int outlineIndex, int pointIndex, Vector3 position)
        {
            selectedShapeOutline[outlineIndex].Insert(pointIndex, ConvertTextureSpaceToSpriteRectSpace(CapPointToRect(position, m_Selected.rect)));
            spriteEditorWindow.SetDataModified();
        }

        public void RemovePointAt(int outlineIndex, int i)
        {
            selectedShapeOutline[outlineIndex].RemoveAt(i);
            spriteEditorWindow.SetDataModified();
        }

        public int GetPointsCount(int outlineIndex)
        {
            return selectedShapeOutline[outlineIndex].Count;
        }

        private Vector2 ConvertSpriteRectSpaceToTextureSpace(Vector2 value)
        {
            Vector2 outlineOffset = new Vector2(0.5f * m_Selected.rect.width + m_Selected.rect.x, 0.5f * m_Selected.rect.height + m_Selected.rect.y);
            value += outlineOffset;
            return value;
        }

        private Vector2 ConvertTextureSpaceToSpriteRectSpace(Vector2 value)
        {
            Vector2 outlineOffset = new Vector2(0.5f * m_Selected.rect.width + m_Selected.rect.x, 0.5f * m_Selected.rect.height + m_Selected.rect.y);
            value -= outlineOffset;
            return value;
        }

        private Vector3 ScreenToLocal(Vector2 point)
        {
            return Handles.inverseMatrix.MultiplyPoint(point);
        }

        private void UndoRedoPerformed()
        {
            shapeEditorDirty = true;
        }

        private void DrawGizmos()
        {
            if (eventSystem.current.type == EventType.Repaint)
            {
                var selected = spriteEditorWindow.selectedSpriteRect;
                if (selected != null)
                {
                    SpriteEditorUtility.BeginLines(styles.spriteBorderColor);
                    SpriteEditorUtility.DrawBox(selected.rect);
                    SpriteEditorUtility.EndLines();
                }
            }
        }

        protected static List<SpriteOutline> GenerateSpriteRectOutline(Rect rect, float detail, byte alphaTolerance, ITextureDataProvider textureProvider, bool useClipper)
        {
            List<SpriteOutline> outline = new List<SpriteOutline>();
            var texture = textureProvider.GetReadableTexture2D();
            if (texture != null)
            {
                Vector2[][] paths;

                // we might have a texture that is capped because of max size or NPOT.
                // in that case, we need to convert values from capped space to actual texture space and back.
                int actualWidth = 0, actualHeight = 0;
                int cappedWidth, cappedHeight;
                textureProvider.GetTextureActualWidthAndHeight(out actualWidth, out actualHeight);
                cappedWidth = texture.width;
                cappedHeight = texture.height;

                Vector2 scale = new Vector2(cappedWidth / (float)actualWidth, cappedHeight / (float)actualHeight);
                Rect spriteRect = rect;
                spriteRect.xMin *= scale.x;
                spriteRect.xMax *= scale.x;
                spriteRect.yMin *= scale.y;
                spriteRect.yMax *= scale.y;

                UnityEditor.Sprites.SpriteUtility.GenerateOutline(texture, spriteRect, detail, alphaTolerance, true, out paths);
                if (useClipper)
                {
                        Clipper2D.Solution clipperSolution = new Clipper2D.Solution();
                        var pathSize = new NativeArray<int>(paths.Length, Allocator.Temp);
                        var pathArguments = new NativeArray<Clipper2D.PathArguments>(paths.Length, Allocator.Temp);
                        var totalPoints = 0;
                        for (int j = 0; j < paths.Length; ++j)
                        {
                            pathSize[j] = paths[j].Length;
                            totalPoints += paths[j].Length;
                            pathArguments[j] = new Clipper2D.PathArguments(Clipper2D.PolyType.ptSubject, true);
                        }
                        var pathPoints= new NativeArray<Vector2>(totalPoints, Allocator.Temp);
                        int pathPointsCounter = 0;
                        for (int j = 0; j < paths.Length; ++j)
                        {
                            NativeArray<Vector2>.Copy(paths[j], 0, pathPoints, pathPointsCounter, paths[j].Length);
                            pathPointsCounter += paths[j].Length;
                        }
                        var executeArgument = new Clipper2D.ExecuteArguments()
                        {
                            initOption = Clipper2D.InitOptions.ioStrictlySimple,
                            clipType = Clipper2D.ClipType.ctUnion,
                            subjFillType = Clipper2D.PolyFillType.pftPositive,
                            clipFillType = Clipper2D.PolyFillType.pftPositive
                        };
                        Clipper2D.Execute(ref clipperSolution, pathPoints, pathSize, pathArguments, executeArgument, Allocator.Temp);
                        paths = new Vector2[clipperSolution.pathSizes.Length][];
                        pathPointsCounter = 0;
                        for (int i = 0; i < paths.Length; ++i)
                        {
                            paths[i] = new Vector2[clipperSolution.pathSizes[i]];
                            NativeArray<Vector2>.Copy(clipperSolution.points, pathPointsCounter, paths[i], 0, paths[i].Length);
                            pathPointsCounter += paths[i].Length;
                        }
                        pathSize.Dispose();
                        pathArguments.Dispose();
                        pathPoints.Dispose();
                        clipperSolution.Dispose();
                }

                Rect capRect = new Rect();
                capRect.size = rect.size;
                capRect.center = Vector2.zero;
                for (int j = 0; j < paths.Length; ++j)
                {
                    SpriteOutline points = new SpriteOutline();
                    foreach (Vector2 v in paths[j])
                        points.Add(CapPointToRect(new Vector2(v.x / scale.x, v.y / scale.y), capRect));

                    outline.Add(points);
                }
            }
            return outline;
        }

        public void Copy()
        {
            if (m_Selected == null || !HasShapeOutline(m_Selected))
                return;

            s_CopyOutline = new SpriteOutlineList(m_Selected.spriteID, m_Outline[m_Selected.spriteID].ToListVectorCapped(m_Selected.rect));
        }

        private void ReplaceOutline(GUID spriteID, List<Vector2[]> newOutline)
        {
            var oldOutline = m_Outline[spriteID];
            m_Outline[spriteID] = new SpriteOutlineList(spriteID, newOutline);
            if (oldOutline != null)
                m_Outline[spriteID].tessellationDetail = oldOutline.tessellationDetail;
        }


        public void Paste()
        {
            if (m_Selected == null || s_CopyOutline == null)
                return;

            RecordUndo();
            ReplaceOutline(m_Selected.spriteID, s_CopyOutline.ToListVectorCapped(m_Selected.rect));
            spriteEditorWindow.SetDataModified();
            shapeEditorDirty = true;
        }

        public void PasteAll()
        {
            if (s_CopyOutline == null)
                return;

            RecordUndo();
            var rectCache = spriteEditorWindow.GetDataProvider<ISpriteEditorDataProvider>().GetSpriteRects();
            if (rectCache != null)
            {
                foreach (var spriteRect in rectCache)
                {
                    var outlines = s_CopyOutline.ToListVectorCapped(spriteRect.rect);
                    ReplaceOutline(spriteRect.spriteID, outlines);
                }
            }
            spriteEditorWindow.SetDataModified();
            shapeEditorDirty = true;
        }

        public void PasteFromAlternate()
        {
            if (m_Selected == null)
                return;

            var alternateOutline = GetAlternateOutlines(m_Selected.spriteID);

            RecordUndo();
            ReplaceOutline(m_Selected.spriteID, alternateOutline);
            spriteEditorWindow.SetDataModified();
            shapeEditorDirty = true;
        }

        public void PasteAllFromAlternate()
        {
            RecordUndo();
            var rectCache = spriteEditorWindow.GetDataProvider<ISpriteEditorDataProvider>().GetSpriteRects();
            if (rectCache != null)
            {
                foreach (var spriteRect in rectCache)
                {
                    var alternateOutline = GetAlternateOutlines(spriteRect.spriteID);
                    ReplaceOutline(spriteRect.spriteID, alternateOutline);
                }
            }
            spriteEditorWindow.SetDataModified();
            shapeEditorDirty = true;
        }

        internal static Vector2 CapPointToRect(Vector2 so, Rect r)
        {
            so.x = Mathf.Min(r.xMax, so.x);
            so.x = Mathf.Max(r.xMin, so.x);
            so.y = Mathf.Min(r.yMax, so.y);
            so.y = Mathf.Max(r.yMin, so.y);
            return so;
        }
    }

    internal class SpriteOutlineModulePreference
    {
        public const string kSettingsUniqueKey = "UnityEditor.U2D.Sprites/SpriteOutlineModule";
        public const string kUseClipper = kSettingsUniqueKey + "kUseClipper";
        public const string kAlphaTolerance = kSettingsUniqueKey + "kAlphaTolerance";
        public const string kPhysicsAlphaTolerance = kSettingsUniqueKey + "kPhysicsAlphaTolerance";

        public static bool useClipper
        {
            get { return EditorPrefs.GetBool(kUseClipper, true); }
            set { EditorPrefs.SetBool(kUseClipper, value); }
        }

        public static int alphaTolerance
        {
            get { return EditorPrefs.GetInt(kAlphaTolerance, 0); }
            set { EditorPrefs.SetInt(kAlphaTolerance, value); }
        }

        public static int physicsAlphaTolerance
        {
            get { return EditorPrefs.GetInt(kPhysicsAlphaTolerance, 200); }
            set { EditorPrefs.SetInt(kPhysicsAlphaTolerance, value); }
        }
    }
}
