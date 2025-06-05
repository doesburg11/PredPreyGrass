using System;
using System.Collections.Generic;
using UnityEngine.Serialization;

namespace UnityEngine.U2D.Animation
{
    [Serializable]
    internal class SpriteCategoryEntryOverride : SpriteCategoryEntry
    {
        [SerializeField]
        bool m_FromMain;
        [SerializeField]
        Sprite m_SpriteOverride;

        public bool fromMain
        {
            get => m_FromMain;
            set => m_FromMain = value;
        }

        public Sprite spriteOverride
        {
            get => m_SpriteOverride;
            set => m_SpriteOverride = value;
        }
    }

    [Serializable]
    internal class SpriteLibCategoryOverride : SpriteLibCategory
    {
        [SerializeField]
        private List<SpriteCategoryEntryOverride> m_OverrideEntries;
        [SerializeField]
        bool m_FromMain;
        [SerializeField]
        int m_EntryOverrideCount;

        public bool fromMain
        {
            get => m_FromMain;
            set => m_FromMain = value;
        }

        public int entryOverrideCount
        {
            get => m_EntryOverrideCount;
            set => m_EntryOverrideCount = value;
        }

        public List<SpriteCategoryEntryOverride> overrideEntries
        {
            get { return m_OverrideEntries; }
            set { m_OverrideEntries = value; }
        }

        public void UpdateOverrideCount()
        {
            // Count only 'new' labels in given category. If it's not from main, then count all categories.
            var overrideCount = 0;
            if (fromMain)
            {
                foreach (var label in overrideEntries)
                {
                    if (!label.fromMain || label.sprite != label.spriteOverride)
                        overrideCount++;
                }
            }
            else
            {
                overrideCount = overrideEntries?.Count ?? 0;
            }

            entryOverrideCount = overrideCount;
        }

        public void RenameDuplicateOverrideEntries()
        {
            if(overrideEntries != null)
                SpriteLibraryAsset.RenameDuplicate(overrideEntries, (_, _) => { });
        }
    }
}