using System;
using System.Collections.Generic;
using System.IO;
using System.Text.RegularExpressions;
using UnityEditorInternal;
using UnityEngine;
using Object = UnityEngine.Object;

namespace UnityEditor.Tilemaps
{
    /// <summary>
    /// A ScriptableObject that stores picks for a particular GridBrushBase type.
    /// The picks include a list of picks defined by the user and a limited list
    /// of picks which were last made by the user. The picks can be loaded onto
    /// the active Brush in the TilePalette.
    /// </summary>
    public class GridBrushPickStore : ScriptableObject
    {
        private static readonly string s_LibraryPickPath = "Library/GridBrush/Pick";
        private static readonly string s_LastPickPath = "Library/GridBrush/Last";
        private static readonly string s_UserPickPath = "Library/GridBrush/User";

        private static readonly string s_LibraryAssetName = "Default";
        private static readonly string s_AssetExtension = ".asset";

        private static readonly string k_GridBrushPickLastSavedMaxCountPref = "GridBrushPickLastSavedMaxCount";
        private static readonly string k_GridBrushPickUserSavedMaxCountPref = "GridBrushPickUserSavedMaxCount";
        private static readonly string k_GridBrushPickLastIndexPref = "GridBrushPickLastIndex";
        private static readonly string k_GridBrushPickLastSavedCountPref = "GridBrushPickLastSavedCount";

        internal static int gridBrushPickLastSavedMaxCount
        {
            get => EditorPrefs.GetInt(k_GridBrushPickLastSavedMaxCountPref, 5);
            set => EditorPrefs.SetInt(k_GridBrushPickLastSavedMaxCountPref, value);
        }

        internal static int gridBrushPickUserSavedMaxCount
        {
            get => EditorPrefs.GetInt(k_GridBrushPickUserSavedMaxCountPref, 50);
            set => EditorPrefs.SetInt(k_GridBrushPickUserSavedMaxCountPref, value);
        }

        internal static int gridBrushPickLastIndex
        {
            get => EditorPrefs.GetInt(k_GridBrushPickLastIndexPref, -1);
            set => EditorPrefs.SetInt(k_GridBrushPickLastIndexPref, value);
        }

        internal static int gridBrushPickLastSavedCount
        {
            get => EditorPrefs.GetInt(k_GridBrushPickLastSavedCountPref, 0);
            set => EditorPrefs.SetInt(k_GridBrushPickLastSavedCountPref, value);
        }

        [HideInInspector]
        [SerializeField]
        private int m_UserSavedCount;
        private Type m_FilteredBrushType;
        [HideInInspector]
        [SerializeField]
        private string m_FilteredBrushText;

        private List<GridBrushBase> m_LastSavedBrushes = new List<GridBrushBase>();

        [SerializeField]
        private List<GridBrushBase> m_UserSavedBrushes = new List<GridBrushBase>();
        private List<GridBrushBase> m_FilteredUserSavedBrushes = new List<GridBrushBase>();

        /// <summary>
        /// The index of the latest last pick that was made.
        /// </summary>
        public int lastIndex => gridBrushPickLastIndex;

        /// <summary>
        /// A list of GridBrushBases which represent the last picks made.
        /// </summary>
        public List<GridBrushBase> lastSavedBrushes
        {
            get => m_LastSavedBrushes;
        }

        /// <summary>
        /// A list of GridBrushBases which represent the user picks made.
        /// </summary>
        public List<GridBrushBase> userSavedBrushes
        {
            get => m_UserSavedBrushes;
        }

        /// <summary>
        /// A list of GridBrushBases which represent the user picks made
        /// filtered by the current filter type.
        /// </summary>
        public List<GridBrushBase> filteredUserSavedBrushes
        {
            get => m_FilteredUserSavedBrushes;
        }

        private void OnDestroy()
        {
            foreach (var brush in m_LastSavedBrushes)
            {
                if (!EditorUtility.IsPersistent(brush))
                    DestroyImmediate(brush);
            }
            m_LastSavedBrushes.Clear();
            foreach (var brush in m_UserSavedBrushes)
            {
                if (!EditorUtility.IsPersistent(brush))
                    DestroyImmediate(brush);
            }
            m_UserSavedBrushes.Clear();
        }

        internal int GetIndexOfLastSavedBrush(GridBrushBase brush)
        {
            return m_LastSavedBrushes.IndexOf(brush);
        }

        internal int GetIndexOfUserBrush(GridBrushBase brush)
        {
            return m_UserSavedBrushes.IndexOf(brush);
        }

        internal int GetIndexOfUserBrushFromFilteredIdx(int filteredIdx)
        {
            if (filteredIdx < 0 || filteredIdx >= filteredUserSavedBrushes.Count)
                return -1;
            return GetIndexOfUserBrush(filteredUserSavedBrushes[filteredIdx]);
        }

        internal bool IsValid()
        {
            return (lastSavedBrushes.Count == 0 || lastSavedBrushes[0] != null)
                   && (userSavedBrushes.Count == 0 || userSavedBrushes[0] != null)
                   && (filteredUserSavedBrushes.Count == 0 || filteredUserSavedBrushes[0] != null);
        }

        /// <summary>
        /// Adds the specified Brush as a new last pick.
        /// </summary>
        /// <description>
        /// This will add a copy of the Brush as a new last pick. If this new copy is over
        /// the limit of the number of last picks, the oldest last pick will be erased.
        /// The last index of the GridBrushPickStore will be changed to this new addition.
        /// </description>
        /// <param name="brush">Brush to save as a new last pick.</param>
        public void AddNewLastSavedBrush(GridBrushBase brush)
        {
            if (brush == null)
                return;

            var clone = Instantiate(brush);
            var name = brush.name;
            if (String.IsNullOrWhiteSpace(name))
                name = brush.GetType().Name;
            clone.name = name;

            var nextIndex = (lastIndex + 1) % gridBrushPickLastSavedMaxCount;
            if (nextIndex < m_LastSavedBrushes.Count)
                m_LastSavedBrushes[nextIndex] = clone;
            else
                m_LastSavedBrushes.Add(clone);
            SaveLibraryGridBrushAsset(clone, nextIndex, false);

            gridBrushPickLastIndex = nextIndex;
            gridBrushPickLastSavedCount = m_LastSavedBrushes.Count;

            GridPaintingState.InvokeBrushPickStoreChanged();
        }

        /// <summary>
        /// Clears the Brush at the index of the last pick list.
        /// </summary>
        /// <description>
        /// This will clear the pick at the index, but will not change the size of the
        /// pick list.
        /// </description>
        /// <param name="index">The index of the Brush of the last pick list to clear.</param>
        public void ClearLastSavedBrush(int index)
        {
            if (index < 0 || index >= m_LastSavedBrushes.Count)
                return;

            m_LastSavedBrushes[index] = null;

            FilterBrushes();
            SaveGridBrushPickStoreAsset();
            var gridBrushPath = GenerateGridBrushInstanceLibraryPath(index, false);
            FileUtil.DeleteFileOrDirectory(gridBrushPath);
        }

        /// <summary>
        /// Adds the specified Brush as a new user pick.
        /// </summary>
        /// <param name="brush">Brush to save as a new user pick.</param>
        public void AddNewUserSavedBrush(GridBrushBase brush)
        {
            if (brush == null)
                return;

            var clone = Instantiate(brush);
            var name = brush.name;
            if (String.IsNullOrWhiteSpace(name))
                name = brush.GetType().Name;
            clone.name = name;
            m_UserSavedBrushes.Add(clone);
            m_UserSavedCount = m_UserSavedBrushes.Count;
            EditorUtility.SetDirty(this);

            FilterBrushes();
            SaveLibraryGridBrushAsset(clone, m_UserSavedCount - 1, true);
            SaveGridBrushPickStoreAsset();
        }

        /// <summary>
        /// Swaps the position of the specified Brush
        /// </summary>
        /// <param name="oldIdx">Index of the Brush to swap.</param>
        /// <param name="newIdx">Index to swap the Brush to.</param>
        public void SwapUserSavedBrushes(int oldIdx, int newIdx)
        {
            if (oldIdx < 0 || oldIdx >= userSavedBrushes.Count)
                return;

            if (newIdx < 0 || newIdx >= userSavedBrushes.Count)
                return;

            var brush = userSavedBrushes[oldIdx];
            userSavedBrushes.RemoveAt(oldIdx);
            userSavedBrushes.Insert(newIdx, brush);

            if (AssetDatabase.IsNativeAsset(this))
            {
                SaveGridBrushPickStoreAsset();
            }
            else
            {
                SaveUserSavedBrushFromIndex(oldIdx < newIdx ? oldIdx : newIdx);
            }
        }

        private void SaveUserSavedBrushFromIndex(int index)
        {
            if (index < 0 || index >= m_UserSavedBrushes.Count)
                return;

            for (int i = index; i < m_UserSavedCount; ++i)
                SaveLibraryGridBrushAsset(m_UserSavedBrushes[i], i, true);

            FilterBrushes();
        }

        /// <summary>
        /// Saves over a brush in the user pick with the given index with
        /// the specified Brush.
        /// </summary>
        /// <param name="index">The index of the Brush of the user pick list to save over.</param>
        /// <param name="brush">Brush to save over as a user pick.</param>
        public void SaveUserSavedBrush(int index, GridBrushBase brush)
        {
            if (brush == null)
                return;

            if (index < 0 || index >= m_UserSavedBrushes.Count)
                return;

            if (m_UserSavedBrushes[index] != brush)
            {
                var clone = Instantiate(brush);
                clone.name = brush.name;
                m_UserSavedBrushes[index] = clone;
                brush = clone;
            }
            m_UserSavedCount = m_UserSavedBrushes.Count;
            EditorUtility.SetDirty(this);

            FilterBrushes();
            SaveLibraryGridBrushAsset(brush, index, true);
            SaveGridBrushPickStoreAsset();
        }

        /// <summary>
        /// Removes the Brush at the index of the last pick list.
        /// </summary>
        /// <param name="index">The index of the Brush of the user pick list to remove.</param>
        /// <returns>Whether the Brush was removed.</returns>
        public bool RemoveUserSavedBrush(int index)
        {
            if (index < 0 || index >= m_UserSavedBrushes.Count)
                return false;

            var brush = m_UserSavedBrushes[index];
            m_UserSavedBrushes.RemoveAt(index);
            m_UserSavedCount = m_UserSavedBrushes.Count;
            EditorUtility.SetDirty(this);

            FilterBrushes();
            SaveGridBrushPickStoreAsset();

            if (brush != null && AssetDatabase.IsNativeAsset(brush))
            {
                AssetDatabase.DeleteAsset(AssetDatabase.GetAssetPath(brush));
            }
            else
            {
                for (int i = index; i < m_UserSavedCount; ++i)
                    SaveLibraryGridBrushAsset(m_UserSavedBrushes[i], i, true);
            }

            return true;
        }

        /// <summary>
        /// Sets the type to filter all user brushes by.
        /// </summary>
        /// <param name="filterType">Type to filter user brushes.</param>
        /// <param name="filterText">Text to filter user brush names.</param>
        public void SetUserBrushFilterType(Type filterType, string filterText)
        {
            m_FilteredBrushType = filterType;
            m_FilteredBrushText = filterText;
            FilterBrushes();
        }

        private void FilterBrushes()
        {
            m_FilteredUserSavedBrushes.Clear();

            foreach (var brush in m_UserSavedBrushes)
            {
                var validBrush = brush != null;
                var hasFilteredBrushType = m_FilteredBrushType != null;
                var hasFilteredBrushText = !String.IsNullOrWhiteSpace(m_FilteredBrushText);

                if (hasFilteredBrushType
                    && validBrush && brush.GetType() != m_FilteredBrushType)
                {
                    continue;
                }
                if (hasFilteredBrushText
                    && validBrush
                    && !Regex.IsMatch(brush.name, m_FilteredBrushText
                        , RegexOptions.Singleline | RegexOptions.IgnoreCase))
                {
                    continue;
                }

                // Show brush only if valid or there is no filter for an invalid brush
                if (validBrush || (!hasFilteredBrushType && !hasFilteredBrushText))
                    m_FilteredUserSavedBrushes.Add(brush);
            }
        }

        private void SaveGridBrushPickStoreAsset()
        {
            if (AssetDatabase.IsNativeAsset(this))
            {
                EditorUtility.SetDirty(this);
                AssetDatabase.SaveAssetIfDirty(this);
            }
            else
            {
                var pickPath = GenerateGridBrushPickLibraryPath(s_LibraryAssetName);
                var folderPath = Path.GetDirectoryName(pickPath);
                if (!Directory.Exists(folderPath))
                {
                    Directory.CreateDirectory(folderPath);
                }
                InternalEditorUtility.SaveToSerializedFileAndForget(new Object[] { this }, pickPath, EditorSettings.serializationMode != SerializationMode.ForceBinary);
            }
            GridPaintingState.InvokeBrushPickStoreChanged();
        }

        internal static GridBrushPickStore LoadOrCreateLibraryGridBrushPickAsset()
        {
            var pickStore = GridBrushPickStoreSettingsProvider.GetUserBrushPickStore();
            if (pickStore == null)
            {
                // Load library Grid Brush Pick Store
                var serializedObjects = InternalEditorUtility.LoadSerializedFileAndForget(GenerateGridBrushPickLibraryPath(s_LibraryAssetName));
                if (serializedObjects != null && serializedObjects.Length > 0)
                {
                    pickStore = serializedObjects[0] as GridBrushPickStore;
                    if (pickStore != null)
                    {
                        var count = pickStore.m_UserSavedCount;
                        var userBrushes = new List<GridBrushBase>();
                        for (int i = 0; i < count; ++i)
                        {
                            var brush = LoadLibraryGridBrushAsset(i, true);
                            if (brush != null)
                            {
                                userBrushes.Add(brush);
                            }
                            else
                            {
                                userBrushes.Add(null);
                            }
                        }
                        pickStore.m_UserSavedBrushes = userBrushes;
                    }
                }
            }

            if (pickStore != null)
            {
                var index = gridBrushPickLastIndex;
                var count = gridBrushPickLastSavedCount;
                var brushes = new List<GridBrushBase>();
                for (int i = 0; i < count; ++i)
                {
                    var brush = LoadLibraryGridBrushAsset(i, false);
                    if (brush != null)
                    {
                        brushes.Add(brush);
                    }
                    else if (index >= brushes.Count)
                    {
                        index--;
                    }
                }
                gridBrushPickLastIndex = index;
                gridBrushPickLastSavedCount = brushes.Count;
                pickStore.m_LastSavedBrushes = brushes;
                pickStore.FilterBrushes();
                return pickStore;
            }

            return CreateLibraryGridBrushPickAsset();
        }

        private void SaveLibraryGridBrushAsset(GridBrushBase brush, int index, bool user)
        {
            if (brush == null)
                return;

            if (user && AssetDatabase.IsNativeAsset(brush))
            {
                var assetPath = AssetDatabase.GetAssetPath(brush);
                var fileName = FileUtil.UnityGetFileNameWithoutExtension(assetPath);
                if (fileName != brush.name)
                {
                    var newName = brush.name;
                    brush.name = fileName;
                    AssetDatabase.SaveAssetIfDirty(brush);
                    AssetDatabase.RenameAsset(assetPath, newName);
                }
                else
                {
                    AssetDatabase.SaveAssetIfDirty(brush);
                }
            }
            else if (user && AssetDatabase.IsNativeAsset(this))
            {
                var assetPath = AssetDatabase.GetAssetPath(this);
                var folderPath = Path.GetDirectoryName(assetPath);
                var gridBrushPath = FileUtil.CombinePaths(folderPath, $"{brush.name}{s_AssetExtension}");
                gridBrushPath = AssetDatabase.GenerateUniqueAssetPath(gridBrushPath);
                AssetDatabase.CreateAsset(brush, gridBrushPath);
            }
            else
            {
                var gridBrushPath = GenerateGridBrushInstanceLibraryPath(index, user);
                var folderPath = Path.GetDirectoryName(gridBrushPath);
                if (!Directory.Exists(folderPath))
                {
                    Directory.CreateDirectory(folderPath);
                }
                InternalEditorUtility.SaveToSerializedFileAndForget(new Object[] { brush }, gridBrushPath, EditorSettings.serializationMode != SerializationMode.ForceBinary);
            }
        }

        private static GridBrushBase LoadLibraryGridBrushAsset(int index, bool user)
        {
            var gridBrushPath = GenerateGridBrushInstanceLibraryPath(index, user);
            var serializedObjects = InternalEditorUtility.LoadSerializedFileAndForget(gridBrushPath);
            if (serializedObjects != null && serializedObjects.Length > 0)
            {
                var brush = serializedObjects[0] as GridBrushBase;
                if (brush != null)
                    return brush;
            }
            return null;
        }

        private static GridBrushPickStore CreateLibraryGridBrushPickAsset()
        {
            var pickStore = CreateInstance<GridBrushPickStore>();
            pickStore.hideFlags = HideFlags.DontSave;
            pickStore.name = s_LibraryAssetName;
            pickStore.SaveGridBrushPickStoreAsset();
            return pickStore;
        }

        private static string GenerateGridBrushPickLibraryPath(string name)
        {
            var path = FileUtil.CombinePaths(s_LibraryPickPath, name + s_AssetExtension);
            path = FileUtil.NiceWinPath(path);
            return path;
        }

        private static string GenerateGridBrushInstanceLibraryPath(int index, bool user)
        {
            var path = FileUtil.CombinePaths(user ? s_UserPickPath : s_LastPickPath, index.ToString() + s_AssetExtension);
            path = FileUtil.NiceWinPath(path);
            return path;
        }
    }
}
