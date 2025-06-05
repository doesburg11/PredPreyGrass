using System;
using System.Collections.Generic;
using UnityEngine;

namespace UnityEditor.U2D.Animation
{
    [Serializable]
    internal class CharacterGroupCache : SkinningObject, ICharacterOrder
    {
        [SerializeField]
        public int parentGroup;
        [SerializeField]
        bool m_IsVisible = true;
        [SerializeField]
        int m_Order = -1;

        public bool isVisible
        {
            get => m_IsVisible;
            set
            {
                m_IsVisible = value;
                skinningCache.GroupVisibilityChanged(this);
            }
        }

        public virtual int order
        {
            get => m_Order;
            set => m_Order = value;
        }
    }

    internal class CharacterPartCache : TransformCache, ICharacterOrder
    {
        [SerializeField]
        SpriteCache m_Sprite;
        [SerializeField]
        List<BoneCache> m_Bones = new List<BoneCache>();
        [SerializeField]
        bool m_IsVisible = true;
        [SerializeField]
        int m_ParentGroup = -1;
        [SerializeField]
        int m_Order = -1;

        public virtual int order
        {
            get => m_Order;
            set => m_Order = value;
        }

        public int parentGroup
        {
            get => m_ParentGroup;
            set => m_ParentGroup = value;
        }

        public virtual bool isVisible
        {
            get => m_IsVisible;
            set
            {
                m_IsVisible = value;
                if (skinningCache != null)
                    skinningCache.SpriteVisibilityChanged(this);
            }
        }

        public int boneCount => m_Bones.Count;

        public virtual SpriteCache sprite
        {
            get => m_Sprite;
            set => m_Sprite = value;
        }

        public virtual BoneCache[] bones
        {
            get => m_Bones.ToArray();
            set => m_Bones = new List<BoneCache>(value);
        }

        public BoneCache GetBone(int index)
        {
            return m_Bones[index];
        }

        public int IndexOf(BoneCache bone)
        {
            return m_Bones.IndexOf(bone);
        }

        public bool Contains(BoneCache bone)
        {
            return m_Bones.Contains(bone);
        }
    }
}
