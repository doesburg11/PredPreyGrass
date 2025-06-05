using System;
using System.Collections;
using System.Linq;
using System.Collections.Generic;
using UnityEngine;
using UnityEditorInternal;

namespace UnityEditor.U2D.Sprites
{
    internal class SpriteRectModel : ScriptableObject, ISerializationCallbackReceiver
    {
        private const float kOverlapTolerance = 0.00001f;
        private const float kBestFitTolerance = 0.5f;

        [Serializable]
        struct StringGUID
        {
            [SerializeField]
            string m_StringGUID;

            public StringGUID(GUID guid)
            {
                m_StringGUID = guid.ToString();
            }

            public static implicit operator GUID(StringGUID d) => new GUID(d.m_StringGUID);
            public static implicit operator StringGUID(GUID d) => new StringGUID(d);
        }

        [Serializable]
        class StringGUIDList : IReadOnlyList<GUID>
        {
            [SerializeField]
            List<StringGUID> m_List = new List<StringGUID>();

            GUID IReadOnlyList<GUID>.this[int index]
            {
                get => m_List[index];
            }

            public StringGUID this[int index]
            {
                get => m_List[index];
                set => m_List[index] = value;
            }

            IEnumerator<GUID> IEnumerable<GUID>.GetEnumerator()
            {
                // Not used for now
                throw new NotImplementedException();
            }

            public int Count => m_List.Count;

            public IEnumerator GetEnumerator()
            {
                return m_List.GetEnumerator();
            }

            public void Clear()
            {
                m_List.Clear();
            }

            public void RemoveAt(int i)
            {
                m_List.RemoveAt(i);
            }

            public void Add(StringGUID value)
            {
                m_List.Add(value);
            }
        }

        /// <summary>
        /// List of all SpriteRects
        /// </summary>
        [SerializeField] private List<SpriteRect> m_SpriteRects;
        /// <summary>
        /// List of all names in the Name-FileId Table
        /// </summary>
        [SerializeField] private List<string> m_SpriteNames;
        /// <summary>
        /// List of all FileIds in the Name-FileId Table
        /// </summary>
        [SerializeField] private StringGUIDList m_SpriteFileIds;
        [SerializeField]
        int m_Version = 0;
        int m_CurrentVersion = 0;

        /// <summary>
        /// HashSet of all names currently in use by SpriteRects
        /// </summary>
        private HashSet<string> m_NamesInUse;
        private HashSet<GUID> m_InternalIdsInUse;

        public IReadOnlyList<SpriteRect> spriteRects => m_SpriteRects;
        public IReadOnlyList<string> spriteNames => m_SpriteNames;
        public IReadOnlyList<GUID> spriteFileIds => m_SpriteFileIds;

        private SpriteRectModel()
        {
            m_SpriteNames = new List<string>();
            m_SpriteFileIds = new StringGUIDList();
            Clear();
        }

        public void RegisterUndo(IUndoSystem undoSystem, string undoMessage)
        {
            undoSystem.RegisterCompleteObjectUndo(this, undoMessage);
            m_CurrentVersion++;
            m_Version = m_CurrentVersion;
        }

        public bool VersionChanged(bool resetVersion)
        {
            var versionChanged = m_Version != m_CurrentVersion;
            if (resetVersion)
                m_CurrentVersion = m_Version;
            return versionChanged;
        }

        public void SetSpriteRects(IList<SpriteRect> newSpriteRects)
        {
            m_SpriteRects.Clear();
            m_SpriteRects.InsertRange(0, newSpriteRects);
            m_NamesInUse = new HashSet<string>();
            m_InternalIdsInUse = new HashSet<GUID>();
            for (var i = 0; i < m_SpriteRects.Count; ++i)
            {
                m_NamesInUse.Add(m_SpriteRects[i].name);
                m_InternalIdsInUse.Add(m_SpriteRects[i].spriteID);
            }
        }

        public void SetNameFileIdPairs(IEnumerable<SpriteNameFileIdPair> pairs)
        {
            m_SpriteNames.Clear();
            m_SpriteFileIds.Clear();

            foreach (var pair in pairs)
                AddNameFileIdPair(pair.name, pair.GetFileGUID());
        }

        public int FindIndex(Predicate<SpriteRect> match)
        {
            int i = 0;
            foreach (var spriteRect in m_SpriteRects)
            {
                if (match.Invoke(spriteRect))
                    return i;
                i++;
            }
            return -1;
        }

        public void Clear()
        {
            m_SpriteRects = new List<SpriteRect>();
            m_NamesInUse = new HashSet<string>();
            m_InternalIdsInUse = new HashSet<GUID>();
        }

        public bool Add(SpriteRect spriteRect, bool shouldReplaceInTable = false)
        {
            if (spriteRect.spriteID.Empty())
            {
                spriteRect.spriteID = GUID.Generate();
            }
            else
            {
                if (IsInternalIdInUsed(spriteRect.spriteID))
                    return false;
            }

            if (shouldReplaceInTable)
            {
                // replace id from sprite to file id table
                if (!UpdateIdInNameIdPair(spriteRect.name, spriteRect.spriteID))
                {
                    // add it into file id table if update wasn't successful i.e. it doesn't exist yet
                    AddNameFileIdPair(spriteRect.name, spriteRect.spriteID);
                }
            }
            else
            {
                // Since we are not replacing the file id table,
                // look for any existing id and set it to the SpriteRect
                var index = m_SpriteNames.FindIndex(x => x == spriteRect.name);
                if (index >= 0)
                {
                    if (IsInternalIdInUsed(m_SpriteFileIds[index]))
                        return false;
                    spriteRect.spriteID = m_SpriteFileIds[index];
                }
                else
                    AddNameFileIdPair(spriteRect.name, spriteRect.spriteID);
            }

            m_SpriteRects.Add(spriteRect);
            m_NamesInUse.Add(spriteRect.name);
            m_InternalIdsInUse.Add(spriteRect.spriteID);
            return true;
        }

        public void Remove(SpriteRect spriteRect)
        {
            m_SpriteRects.Remove(spriteRect);
            m_NamesInUse.Remove(spriteRect.name);
            m_InternalIdsInUse.Remove(spriteRect.spriteID);
        }

        /// <summary>
        /// Checks whether or not the name is currently in use by any of the SpriteRects in the texture.
        /// </summary>
        /// <param name="rectName">The name to check for</param>
        /// <returns>True if the name is currently in use</returns>
        public bool IsNameUsed(string rectName)
        {
            return m_NamesInUse.Contains(rectName);
        }

        /// <summary>
        /// Checks whether or not the id is currently in use by any of the SpriteRects in the texture.
        /// </summary>
        /// <param name="rectName">The id to check for</param>
        /// <returns>True if the name is currently in use</returns>
        public bool IsInternalIdInUsed(GUID internalId)
        {
            return m_InternalIdsInUse.Contains(internalId);
        }

        public List<SpriteRect> GetSpriteRects()
        {
            return m_SpriteRects;
        }

        public bool Rename(string oldName, string newName, GUID fileId)
        {
            if (!IsNameUsed(oldName))
                return false;
            if (IsNameUsed(newName))
                return false;

            var index = m_SpriteNames.FindIndex(x => x == oldName);
            if (index >= 0)
            {
                m_SpriteNames.RemoveAt(index);
                m_SpriteFileIds.RemoveAt(index);
            }

            index = m_SpriteNames.FindIndex(x => x == newName);
            if (index >= 0)
                m_SpriteFileIds[index] = fileId;
            else
                AddNameFileIdPair(newName, fileId);

            m_NamesInUse.Remove(oldName);
            m_NamesInUse.Add(newName);
            return true;
        }

        void AddNameFileIdPair(string spriteName, GUID fileId)
        {
            m_SpriteNames.Add(spriteName);
            m_SpriteFileIds.Add(fileId);
        }

        bool UpdateIdInNameIdPair(string spriteName, GUID newFileId)
        {
            var index = m_SpriteNames.FindIndex(x => x == spriteName);
            if (index >= 0)
            {
                m_SpriteFileIds[index] = newFileId;
                return true;
            }

            return false;
        }

        public void ClearUnusedFileID()
        {
            m_SpriteNames.Clear();
            m_SpriteFileIds.Clear();
            foreach (var sprite in m_SpriteRects)
            {
                m_SpriteNames.Add(sprite.name);
                m_SpriteFileIds.Add(sprite.spriteID);
            }
        }

        public int AddSprite(Rect rect, int alignment, Vector2 pivot, string name, Vector4 border)
        {
            if (IsNameUsed(name))
                return -1;

            SpriteRect spriteRect = new SpriteRect();
            spriteRect.rect = rect;
            spriteRect.alignment = (SpriteAlignment)alignment;
            spriteRect.pivot = pivot;
            spriteRect.name = name;
            spriteRect.originalName = spriteRect.name;
            spriteRect.border = border;

            spriteRect.spriteID = GUID.Generate();

            if (!Add(spriteRect))
                return -1;
            return spriteRects.Count - 1;
        }

        public int AddSprite(Rect frame, int alignment, Vector2 pivot, SpriteFrameModule.AutoSlicingMethod slicingMethod, int originalCount, ref int nameIndex, Func<int, string> nameGenerate)
        {
            int outSprite = -1;
            switch (slicingMethod)
            {
                case SpriteFrameModule.AutoSlicingMethod.DeleteAll:
                    {
                        while (outSprite == -1)
                        {
                            outSprite = AddSprite(frame, alignment, pivot, nameGenerate(nameIndex++), Vector4.zero);
                        }
                    }
                    break;
                case SpriteFrameModule.AutoSlicingMethod.Smart:
                    {
                        outSprite = GetExistingOverlappingSprite(frame, originalCount, true);
                        if (outSprite != -1)
                        {
                            var existingRect = spriteRects[outSprite];
                            existingRect.rect = frame;
                            existingRect.alignment = (SpriteAlignment)alignment;
                            existingRect.pivot = pivot;
                        }
                        else
                        {
                            while (outSprite == -1)
                            {
                                outSprite = AddSprite(frame, alignment, pivot, nameGenerate(nameIndex++), Vector4.zero);
                            }
                        }
                    }
                    break;
                case SpriteFrameModule.AutoSlicingMethod.Safe:
                    {
                        outSprite = GetExistingOverlappingSprite(frame, originalCount);
                        while (outSprite == -1)
                        {
                            outSprite = AddSprite(frame, alignment, pivot, nameGenerate(nameIndex++), Vector4.zero);
                        }
                    }
                    break;
            }
            return outSprite;
        }

        private int GetExistingOverlappingSprite(Rect rect, int originalCount, bool bestFit = false)
        {
            var count = Math.Min(originalCount, spriteRects.Count);
            var bestRect = -1;
            var rectArea = rect.width * rect.height;
            if (rectArea < kOverlapTolerance)
                return bestRect;

            var bestRatio = float.MaxValue;
            var bestArea = float.MaxValue;
            for (int i = 0; i < count; i++)
            {
                Rect existingRect = spriteRects[i].rect;
                if (existingRect.Overlaps(rect))
                {
                    if (bestFit)
                    {
                        var dx = Math.Min(rect.xMax, existingRect.xMax) - Math.Max(rect.xMin, existingRect.xMin);
                        var dy = Math.Min(rect.yMax, existingRect.yMax) - Math.Max(rect.yMin, existingRect.yMin);
                        var overlapArea = dx * dy;
                        var overlapRatio = Math.Abs((overlapArea / rectArea) - 1.0f);
                        var existingArea = existingRect.width * existingRect.height;
                        if (overlapRatio < bestRatio || (overlapRatio < kOverlapTolerance && existingArea < bestArea))
                        {
                            bestRatio = overlapRatio;
                            if (overlapRatio < kOverlapTolerance)
                                bestArea = existingArea;
                            bestRect = i;
                        }
                    }
                    else
                    {
                        bestRect = i;
                        break;
                    }
                }
            }
            if (bestFit && bestRatio > kBestFitTolerance)
                return -1;
            return bestRect;
        }

        void ISerializationCallbackReceiver.OnBeforeSerialize()
        {}

        void ISerializationCallbackReceiver.OnAfterDeserialize()
        {
            SetSpriteRects(new List<SpriteRect>(m_SpriteRects));
        }
    }

    internal class OutlineSpriteRect : SpriteRect
    {
        public List<Vector2[]> outlines;

        public OutlineSpriteRect(SpriteRect rect)
        {
            this.name = rect.name;
            this.originalName = rect.originalName;
            this.pivot = rect.pivot;
            this.alignment = rect.alignment;
            this.border = rect.border;
            this.rect = rect.rect;
            this.spriteID = rect.spriteID;
            outlines = new List<Vector2[]>();
        }
    }

    [Serializable]
    struct EditCapabilityUndoData
    {
        public EditCapability data;
        public EditCapability originalData;
    }

    [Serializable]
    class EditCapabilityUndoObject : ScriptableObject
    {
        [SerializeField]
        EditCapabilityUndoData m_Data;

        [SerializeField]
        int m_Version = 0;
        int m_CurrentVersion = 0;

        public static EditCapabilityUndoObject CreateInstance(EditCapability data)
        {
            var undoObject = CreateInstance<EditCapabilityUndoObject>();
            undoObject.hideFlags = HideFlags.HideAndDontSave;
            undoObject.Init(data);
            return undoObject;
        }

        public void Init(EditCapability data)
        {
            m_Data = new EditCapabilityUndoData()
            {
                data = data,
                originalData = data
            };
        }

        public string SaveData()
        {
            return JsonUtility.ToJson(m_Data);
        }

        public bool LoadData(string data)
        {
            try
            {
                var previous = JsonUtility.FromJson<EditCapabilityUndoData>(data);
                m_Data.data = previous.data;
                return true;
            }
            catch (Exception)
            {
                // nothing to do here.
            }

            return false;

        }

        public void RegisterUndo(IUndoSystem undoSystem, EditCapability data, string undoMessage)
        {
            if (!data.Equals(m_Data))
            {
                undoSystem.RegisterCompleteObjectUndo(this, undoMessage);
                m_Data.data = data;
                m_CurrentVersion++;
                m_Version = m_CurrentVersion;
            }
        }

        public void SetData(EditCapability data)
        {
            m_Data.data = data;
        }

        public EditCapability data => m_Data.data;
        public EditCapability originalData => m_Data.originalData;

        public bool VersionChanged(bool resetVersion)
        {
            bool returnValue = m_CurrentVersion != m_Version;
            if (resetVersion)
                m_CurrentVersion = m_Version;
            return returnValue;
        }

        public void Dispose()
        {
            UnityEditor.Undo.ClearUndo(this);
        }
    }

    internal abstract partial class SpriteFrameModuleBase : SpriteEditorModuleModeSupportBase
    {
        [Serializable]
        internal class SpriteFrameModulePersistentState : ScriptableSingleton<SpriteFrameModulePersistentState>
        {
            public PivotUnitMode pivotUnitMode = PivotUnitMode.Normalized;
        }

        protected SpriteRectModel m_RectsCache;
        protected ITextureDataProvider m_TextureDataProvider;
        protected ISpriteEditorDataProvider m_SpriteDataProvider;
        protected ISpriteNameFileIdDataProvider m_NameFileIdDataProvider;
        protected ISpriteCustomDataProvider m_CustomDataProvider;
        protected ISpriteFrameEditCapability m_FrameEditCapability;
        string m_ModuleName;
        protected EditCapabilityUndoObject m_CurrentEditEditCapability;
        protected event Action m_OnUndoCallback;
        bool m_Undoing = false;

        internal enum PivotUnitMode
        {
            Normalized,
            Pixels
        }

        static PivotUnitMode pivotUnitMode
        {
            get => SpriteFrameModulePersistentState.instance.pivotUnitMode;
            set => SpriteFrameModulePersistentState.instance.pivotUnitMode = value;
        }

        protected SpriteFrameModuleBase(string name, ISpriteEditor sw, IEventSystem es, IUndoSystem us, IAssetDatabase ad)
        {
            spriteEditor = sw;
            eventSystem = es;
            undoSystem = us;
            assetDatabase = ad;
            m_ModuleName = name;
        }

        // implements ISpriteEditorModule

        public override void OnModuleActivate()
        {
            m_SpriteDataProvider = GetDataProvider<ISpriteEditorDataProvider>();
            spriteImportMode = SpriteFrameModule.GetSpriteImportMode(m_SpriteDataProvider);
            m_TextureDataProvider = GetDataProvider<ITextureDataProvider>();
            m_NameFileIdDataProvider = GetDataProvider<ISpriteNameFileIdDataProvider>();
            m_CustomDataProvider = spriteEditor.GetDataProvider<ISpriteCustomDataProvider>();
            m_FrameEditCapability = GetDataProvider<ISpriteFrameEditCapability>();

            InitEditCapabilityUndoObject();
            m_TextureDataProvider.RegisterDataChangeCallback(OnTextureDataProviderChanged);
            OnTextureDataProviderChanged(m_TextureDataProvider);
            InitSpriteRectCache();

            AddMainUI(spriteEditor.GetMainVisualContainer());
            undoSystem.RegisterUndoCallback(UndoCallback);
            foreach (var mode in modes)
            {
                mode.OnAddToModule(this);
            }
        }

        void InitEditCapabilityUndoObject()
        {
            if (m_CurrentEditEditCapability != null)
            {
                undoSystem.ClearUndo(m_CurrentEditEditCapability);
                ScriptableObject.DestroyImmediate(m_CurrentEditEditCapability);
            }
            var capability = m_FrameEditCapability?.GetEditCapability() ??EditCapability.defaultCapability;
            m_CurrentEditEditCapability = EditCapabilityUndoObject.CreateInstance(capability);
        }

        void OnTextureDataProviderChanged(ITextureDataProvider obj)
        {
            int width, height;
            m_TextureDataProvider.GetTextureActualWidthAndHeight(out width, out height);
            textureActualWidth = width;
            textureActualHeight = height;
        }

        void InitSpriteRectCache()
        {
            if (m_RectsCache != null)
            {
                undoSystem.ClearUndo(m_RectsCache);
                ScriptableObject.DestroyImmediate(m_RectsCache);
            }
            var spriteList = m_SpriteDataProvider.GetSpriteRects().ToList();
            if (m_NameFileIdDataProvider == null)
                m_NameFileIdDataProvider = new DefaultSpriteNameFileIdDataProvider(spriteList);
            var nameFileIdPairs = m_NameFileIdDataProvider.GetNameFileIdPairs();

            m_RectsCache = ScriptableObject.CreateInstance<SpriteRectModel>();
            m_RectsCache.hideFlags = HideFlags.HideAndDontSave;

            m_RectsCache.SetSpriteRects(spriteList);
            spriteEditor.spriteRects = spriteList;
            m_RectsCache.SetNameFileIdPairs(nameFileIdPairs);

            if (spriteEditor.selectedSpriteRect != null)
                spriteEditor.selectedSpriteRect = m_RectsCache.spriteRects.FirstOrDefault(x => x.spriteID == spriteEditor.selectedSpriteRect.spriteID);
        }

        public override void OnModuleDeactivate()
        {
            foreach (var mode in modes)
            {
                mode.OnRemoveFromModule(this);
            }
            if (m_RectsCache != null)
            {
                undoSystem.ClearUndo(m_RectsCache);
                ScriptableObject.DestroyImmediate(m_RectsCache);
                spriteEditor.spriteRects = m_SpriteDataProvider.GetSpriteRects().ToList();
                m_RectsCache = null;
            }

            if (m_CurrentEditEditCapability != null)
            {
                undoSystem.ClearUndo(m_CurrentEditEditCapability);
                ScriptableObject.DestroyImmediate(m_CurrentEditEditCapability);
                m_CurrentEditEditCapability = null;
            }

            m_TextureDataProvider.UnregisterDataChangeCallback(OnTextureDataProviderChanged);
            undoSystem.UnregisterUndoCallback(UndoCallback);
            RemoveMainUI(spriteEditor.GetMainVisualContainer());
        }

        protected void OnEditCapabilityChanged(EEditCapability arg1, bool arg2)
        {
            var data = m_CurrentEditEditCapability.data;
            data.SetCapability(arg1, arg2);
            if (!m_Undoing)
                m_CurrentEditEditCapability.RegisterUndo(undoSystem, data, "Change Edit Capability");
            else
                m_CurrentEditEditCapability.SetData(data);
            spriteEditor.RequestRepaint();
            PopulateSpriteFrameInspectorField();
        }

        public override bool ApplyRevert(bool apply)
        {
            if (apply)
            {
                var array = m_RectsCache != null ? m_RectsCache.spriteRects.ToArray() : null;
                var spriteDataProvider = spriteEditor.GetDataProvider<ISpriteEditorDataProvider>();
                var nameFileIdDataProvider = spriteEditor.GetDataProvider<ISpriteNameFileIdDataProvider>();
                spriteDataProvider.SetSpriteRects(array);

                var spriteNames = m_RectsCache?.spriteNames;
                var spriteFileIds = m_RectsCache?.spriteFileIds;
                if (spriteNames != null && spriteFileIds != null && nameFileIdDataProvider != null)
                {
                    var pairList = new List<SpriteNameFileIdPair>(spriteNames.Count);
                    for (var i = 0; i < spriteNames.Count; ++i)
                        pairList.Add(new SpriteNameFileIdPair(spriteNames[i], spriteFileIds[i]));
                    nameFileIdDataProvider.SetNameFileIdPairs(pairList.ToArray());
                }

                var outlineDataProvider = spriteDataProvider.GetDataProvider<ISpriteOutlineDataProvider>();
                var physicsDataProvider = spriteDataProvider.GetDataProvider<ISpritePhysicsOutlineDataProvider>();
                foreach (var rect in array)
                {
                    if (rect is OutlineSpriteRect outlineRect)
                    {
                        if (outlineRect.outlines.Count > 0)
                        {
                            outlineDataProvider.SetOutlines(outlineRect.spriteID, outlineRect.outlines);
                            physicsDataProvider.SetOutlines(outlineRect.spriteID, outlineRect.outlines);
                        }
                    }
                }

                if (m_RectsCache != null)
                    undoSystem.ClearUndo(m_RectsCache);

                if (m_CurrentEditEditCapability != null)
                    undoSystem.ClearUndo(m_CurrentEditEditCapability);
            }
            else
            {
                    InitSpriteRectCache();
                InitEditCapabilityUndoObject();
            }

            return true;
        }

        public override string moduleName
        {
            get { return m_ModuleName; }
        }

        // injected interfaces
        protected IEventSystem eventSystem
        {
            get;
            private set;
        }

        protected IUndoSystem undoSystem
        {
            get;
            private set;
        }

        protected IAssetDatabase assetDatabase
        {
            get;
            private set;
        }

        protected SpriteRect selected
        {
            get { return spriteEditor.selectedSpriteRect; }
            set { spriteEditor.selectedSpriteRect = value; }
        }

        protected SpriteImportMode spriteImportMode
        {
            get; private set;
        }

        protected string spriteAssetPath
        {
            get { return assetDatabase.GetAssetPath(m_SpriteDataProvider.targetObject); }
        }

        public bool hasSelected
        {
            get { return spriteEditor.selectedSpriteRect != null; }
        }

        public SpriteAlignment selectedSpriteAlignment
        {
            get { return selected.alignment; }
        }

        public Vector2 selectedSpritePivot
        {
            get { return selected.pivot; }
        }

        private Vector2 selectedSpritePivotInCurUnitMode
        {
            get
            {
                return pivotUnitMode == PivotUnitMode.Pixels
                    ? ConvertFromNormalizedToRectSpace(selectedSpritePivot, selectedSpriteRect_Rect)
                    : selectedSpritePivot;
            }
        }

        public int CurrentSelectedSpriteIndex()
        {
            if (m_RectsCache != null && selected != null)
                return m_RectsCache.FindIndex(x => x.spriteID == selected.spriteID);
            return -1;
        }

        public Vector4 selectedSpriteBorder
        {
            get { return ClampSpriteBorderToRect(selected.border, selected.rect); }
            set
            {
                m_RectsCache.RegisterUndo(undoSystem, "Change Sprite Border");
                selected.border = ClampSpriteBorderToRect(value, selected.rect);
                NotifyOnSpriteRectChanged();
                spriteEditor.SetDataModified();
            }
        }

        public Rect selectedSpriteRect_Rect
        {
            get { return selected.rect; }
            set
            {
                m_RectsCache.RegisterUndo(undoSystem, "Change Sprite rect");
                selected.rect = ClampSpriteRect(value, textureActualWidth, textureActualHeight);
                NotifyOnSpriteRectChanged();
                spriteEditor.SetDataModified();
            }
        }

        public string selectedSpriteName
        {
            get { return selected.name; }
            set
            {
                if (selected.name == value)
                    return;
                if (m_RectsCache.IsNameUsed(value))
                    return;

                string oldName = selected.name;
                string newName = InternalEditorUtility.RemoveInvalidCharsFromFileName(value, true);

                // These can only be changed in sprite multiple mode
                if (string.IsNullOrEmpty(selected.originalName) && (newName != oldName))
                    selected.originalName = oldName;

                // Is the name empty?
                if (string.IsNullOrEmpty(newName))
                    newName = oldName;

                // Did the rename succeed?
                if (m_RectsCache.Rename(oldName, newName, selected.spriteID))
                {
                    m_RectsCache.RegisterUndo(undoSystem, "Change Sprite Name");
                    selected.name = newName;
                    NotifyOnSpriteRectChanged();
                    spriteEditor.SetDataModified();
                }
            }
        }

        public int spriteCount
        {
            get { return m_RectsCache.spriteRects.Count; }
        }

        public Vector4 GetSpriteBorderAt(int i)
        {
            return m_RectsCache.spriteRects[i].border;
        }

        public Rect GetSpriteRectAt(int i)
        {
            return m_RectsCache.spriteRects[i].rect;
        }

        public int textureActualWidth { get; private set; }
        public int textureActualHeight { get; private set; }

        public void SetSpritePivotAndAlignment(Vector2 pivot, SpriteAlignment alignment)
        {
            m_RectsCache.RegisterUndo(undoSystem, "Change Sprite Pivot");
            selected.alignment = alignment;
            selected.pivot = SpriteEditorUtility.GetPivotValue(alignment, pivot);
            NotifyOnSpriteRectChanged();
            spriteEditor.SetDataModified();
        }

        public bool containsMultipleSprites
        {
            get { return spriteImportMode == SpriteImportMode.Multiple; }
        }

        protected void SnapPivotToSnapPoints(Vector2 pivot, out Vector2 outPivot, out SpriteAlignment outAlignment)
        {
            Rect rect = selectedSpriteRect_Rect;

            // Convert from normalized space to texture space
            Vector2 texturePos = new Vector2(rect.xMin + rect.width * pivot.x, rect.yMin + rect.height * pivot.y);

            Vector2[] snapPoints = GetSnapPointsArray(rect);

            // Snapping is now a firm action, it will always snap to one of the snapping points.
            SpriteAlignment snappedAlignment = SpriteAlignment.Custom;
            float nearestDistance = float.MaxValue;
            for (int alignment = 0; alignment < snapPoints.Length; alignment++)
            {
                float distance = (texturePos - snapPoints[alignment]).magnitude * m_Zoom;
                if (distance < nearestDistance)
                {
                    snappedAlignment = (SpriteAlignment)alignment;
                    nearestDistance = distance;
                }
            }

            outAlignment = snappedAlignment;
            outPivot = ConvertFromTextureToNormalizedSpace(snapPoints[(int)snappedAlignment], rect);
        }

        protected void SnapPivotToPixels(Vector2 pivot, out Vector2 outPivot, out SpriteAlignment outAlignment)
        {
            outAlignment = SpriteAlignment.Custom;

            Rect rect = selectedSpriteRect_Rect;
            float unitsPerPixelX = 1.0f / rect.width;
            float unitsPerPixelY = 1.0f / rect.height;
            outPivot.x = Mathf.Round(pivot.x / unitsPerPixelX) * unitsPerPixelX;
            outPivot.y = Mathf.Round(pivot.y / unitsPerPixelY) * unitsPerPixelY;
        }

        private void UndoCallback()
        {
            m_Undoing = true;
            if(m_RectsCache.VersionChanged(true))
                NotifyOnSpriteRectChanged();
            if (m_CurrentEditEditCapability.VersionChanged(true))
            {
                m_OnUndoCallback?.Invoke();
                PopulateSpriteFrameInspectorField();
            }

            UIUndoCallback();
            m_Undoing = false;
        }

        protected static Rect ClampSpriteRect(Rect rect, float maxX, float maxY)
        {
            // Clamp rect to width height
            rect = FlipNegativeRect(rect);
            Rect newRect = new Rect();

            newRect.xMin = Mathf.Clamp(rect.xMin, 0, maxX - 1);
            newRect.yMin = Mathf.Clamp(rect.yMin, 0, maxY - 1);
            newRect.xMax = Mathf.Clamp(rect.xMax, 1, maxX);
            newRect.yMax = Mathf.Clamp(rect.yMax, 1, maxY);

            // Prevent width and height to be 0 value after clamping.
            if (Mathf.RoundToInt(newRect.width) == 0)
                newRect.width = 1;
            if (Mathf.RoundToInt(newRect.height) == 0)
                newRect.height = 1;

            return SpriteEditorUtility.RoundedRect(newRect);
        }

        protected static Rect FlipNegativeRect(Rect rect)
        {
            Rect newRect = new Rect();

            newRect.xMin = Mathf.Min(rect.xMin, rect.xMax);
            newRect.yMin = Mathf.Min(rect.yMin, rect.yMax);
            newRect.xMax = Mathf.Max(rect.xMin, rect.xMax);
            newRect.yMax = Mathf.Max(rect.yMin, rect.yMax);

            return newRect;
        }

        protected static Vector4 ClampSpriteBorderToRect(Vector4 border, Rect rect)
        {
            Rect flipRect = FlipNegativeRect(rect);
            float w = flipRect.width;
            float h = flipRect.height;

            Vector4 newBorder = new Vector4();

            // Make sure borders are within the width/height and left < right and top < bottom
            newBorder.x = Mathf.RoundToInt(Mathf.Clamp(border.x, 0, Mathf.Min(Mathf.Abs(w - border.z), w))); // Left
            newBorder.z = Mathf.RoundToInt(Mathf.Clamp(border.z, 0, Mathf.Min(Mathf.Abs(w - newBorder.x), w))); // Right

            newBorder.y = Mathf.RoundToInt(Mathf.Clamp(border.y, 0, Mathf.Min(Mathf.Abs(h - border.w), h))); // Bottom
            newBorder.w = Mathf.RoundToInt(Mathf.Clamp(border.w, 0, Mathf.Min(Mathf.Abs(h - newBorder.y), h))); // Top

            return newBorder;
        }

        protected virtual void NotifyOnSpriteRectChanged() { }
    }
}
