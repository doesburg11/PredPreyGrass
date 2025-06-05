using System;
using System.Collections.Generic;
using UnityEditor.U2D.Sprites;
using UnityEngine;

namespace UnityEditor.U2D.Animation
{
    [Serializable]
    internal class MeshCache : BaseSpriteMeshData
    {
        [SerializeField]
        List<BoneCache> m_Bones = new List<BoneCache>();
        [SerializeField]
        SpriteCache m_Sprite;

        public override string spriteName => sprite.name;
        public override int boneCount => m_Bones.Count;
        public override Rect frame => sprite.textureRect;

        public ITextureDataProvider textureDataProvider { get; set; }

        public SpriteCache sprite
        {
            get => m_Sprite;
            set => m_Sprite = value;
        }

        public BoneCache[] bones
        {
            get => m_Bones.ToArray();
            set => SetBones(value);
        }

        public override SpriteBoneData GetBoneData(int index)
        {
            var worldToLocalMatrix = sprite.worldToLocalMatrix;

            //We expect m_Bones to contain character's bones references if character exists. Sprite's skeleton bones otherwise.
            if (sprite.skinningCache.hasCharacter)
                worldToLocalMatrix = sprite.GetCharacterPart().worldToLocalMatrix;

            SpriteBoneData spriteBoneData;
            var bone = m_Bones[index];

            if (bone == null)
                spriteBoneData = new SpriteBoneData();
            else
            {
                spriteBoneData = new SpriteBoneData()
                {
                    parentId = bone.parentBone == null ? -1 : m_Bones.IndexOf(bone.parentBone),
                    localPosition = bone.localPosition,
                    localRotation = bone.localRotation,
                    position = worldToLocalMatrix.MultiplyPoint3x4(bone.position),
                    endPosition = worldToLocalMatrix.MultiplyPoint3x4(bone.endPosition),
                    depth = bone.depth,
                    length = bone.localLength
                };
            }

            return spriteBoneData;
        }

        public override float GetBoneDepth(int index)
        {
            return m_Bones[index].depth;
        }

        public bool ContainsBone(BoneCache bone)
        {
            return m_Bones.Contains(bone);
        }

        public void SetCompatibleBoneSet(BoneCache[] boneCache)
        {
            m_Bones = new List<BoneCache>(boneCache);
        }

        void SetBones(BoneCache[] boneCache)
        {
            FixWeights(boneCache);
            SetCompatibleBoneSet(boneCache);
        }

        void FixWeights(BoneCache[] newBones)
        {
            var newBonesList = new List<BoneCache>(newBones);
            var indexMap = new Dictionary<int, int>();

            for (var i = 0; i < m_Bones.Count; ++i)
            {
                var bone = m_Bones[i];
                var newIndex = newBonesList.IndexOf(bone);

                if (newIndex != -1)
                    indexMap.Add(i, newIndex);
            }

            for (var i = 0; i < vertexWeights.Length; ++i)
            {
                var boneWeight = vertexWeights[i];
                for (var m = 0; m < boneWeight.Count; ++m)
                {
                    var boneRemoved = indexMap.TryGetValue(boneWeight[m].boneIndex, out var newIndex) == false;

                    if (boneRemoved)
                    {
                        boneWeight[m].weight = 0f;
                        boneWeight[m].enabled = false;
                    }

                    boneWeight[m].boneIndex = newIndex;

                    if (boneRemoved)
                        boneWeight.CompensateOtherChannels(m);
                }

                boneWeight.UnifyChannelsWithSameBoneIndex();
                vertexWeights[i] = boneWeight;
            }
        }
    }
}
