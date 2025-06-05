using System.Collections.Generic;

namespace UnityEngine.U2D.Animation
{
    internal class SpriteLibrarySourceAsset : ScriptableObject
    {
        public const string defaultName = "New Sprite Library Asset";
        public const string extension = ".spriteLib";

        [SerializeField]
        List<SpriteLibCategoryOverride> m_Library = new();

        [SerializeField]
        string m_PrimaryLibraryGUID;

        [SerializeField]
        long m_ModificationHash;

        [SerializeField]
        int m_Version = 1;

        public IReadOnlyList<SpriteLibCategoryOverride> library => m_Library;

        public string primaryLibraryGUID => m_PrimaryLibraryGUID;

        public long modificationHash => m_ModificationHash;
        public int version => m_Version;

        public void InitializeWithAsset(SpriteLibrarySourceAsset source)
        {
            m_Library = new List<SpriteLibCategoryOverride>(source.m_Library);
            m_PrimaryLibraryGUID = source.m_PrimaryLibraryGUID;
            m_ModificationHash = source.m_ModificationHash;
        }

        public void SetLibrary(IList<SpriteLibCategoryOverride> newLibrary)
        {
            if (!m_Library.Equals(newLibrary))
            {
                m_Library = new List<SpriteLibCategoryOverride>(newLibrary);
                UpdateModificationHash();
            }
        }

        public void SetPrimaryLibraryGUID(string newPrimaryLibraryGUID)
        {
            if (m_PrimaryLibraryGUID != newPrimaryLibraryGUID)
            {
                m_PrimaryLibraryGUID = newPrimaryLibraryGUID;
                UpdateModificationHash();
            }
        }

        public void AddCategory(SpriteLibCategoryOverride newCategory)
        {
            if (!m_Library.Contains(newCategory))
            {
                m_Library.Add(newCategory);
                UpdateModificationHash();
            }
        }

        public void RemoveCategory(SpriteLibCategoryOverride categoryToRemove)
        {
            if (m_Library.Contains(categoryToRemove))
            {
                m_Library.Remove(categoryToRemove);
                UpdateModificationHash();
            }
        }

        public void ClearCategories()
        {
            m_Library.Clear();
        }

        public void RemoveCategory(int indexToRemove)
        {
            if (indexToRemove >= 0 && indexToRemove < m_Library.Count)
            {
                m_Library.RemoveAt(indexToRemove);
                UpdateModificationHash();
            }
        }

        void UpdateModificationHash()
        {
            m_ModificationHash = SpriteLibraryUtility.GenerateHash();
        }
    }
}