using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace UnityEditor.U2D.Animation
{
    internal class SkeletonCache : TransformCache
    {
        [SerializeField]
        bool m_IsPosePreview = false;
        [SerializeField]
        List<BoneCache> m_Bones = new List<BoneCache>();

        public bool isPosePreview => m_IsPosePreview;

        public int boneCount => m_Bones.Count;

        public virtual BoneCache[] bones => m_Bones.ToArray();

        public void AddBone(BoneCache bone, bool worldPositionStays = true)
        {
            Debug.Assert(bone != null);
            Debug.Assert(!Contains(bone));

            if (bone.parent == null)
                bone.SetParent(this, worldPositionStays);

            m_Bones.Add(bone);
        }

        public void ReorderBones(IEnumerable<BoneCache> boneCache)
        {
            if (boneCache.Count() == m_Bones.Count)
            {
                foreach (var b in m_Bones)
                {
                    if (!boneCache.Contains(b))
                        return;
                }

                m_Bones = boneCache.ToList();
            }
        }

        public void DestroyBone(BoneCache bone)
        {
            Debug.Assert(bone != null);
            Debug.Assert(Contains(bone));

            m_Bones.Remove(bone);

            var boneChildren = bone.children;
            foreach (var child in boneChildren)
                child.SetParent(bone.parent);

            skinningCache.Destroy(bone);
        }

        public void SetDefaultPose()
        {
            foreach (var bone in m_Bones)
                bone.SetDefaultPose();

            m_IsPosePreview = false;
        }

        public void RestoreDefaultPose()
        {
            foreach (var bone in m_Bones)
                bone.RestoreDefaultPose();

            m_IsPosePreview = false;
            skinningCache.events.skeletonPreviewPoseChanged.Invoke(this);
        }

        public void SetPosePreview()
        {
            m_IsPosePreview = true;
        }

        public BonePose[] GetLocalPose()
        {
            var pose = new List<BonePose>();

            foreach (var bone in m_Bones)
                pose.Add(bone.localPose);

            return pose.ToArray();
        }

        public void SetLocalPose(BonePose[] pose)
        {
            Debug.Assert(m_Bones.Count == pose.Length);

            for (var i = 0; i < m_Bones.Count; ++i)
                m_Bones[i].localPose = pose[i];

            m_IsPosePreview = true;
        }

        public BonePose[] GetWorldPose()
        {
            var pose = new List<BonePose>();

            foreach (var bone in m_Bones)
                pose.Add(bone.worldPose);

            return pose.ToArray();
        }

        public void SetWorldPose(BonePose[] pose)
        {
            Debug.Assert(m_Bones.Count == pose.Length);

            for (var i = 0; i < m_Bones.Count; ++i)
            {
                var bone = m_Bones[i];
                var childWoldPose = bone.GetChildrenWoldPose();
                bone.worldPose = pose[i];
                bone.SetChildrenWorldPose(childWoldPose);
            }

            m_IsPosePreview = true;
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

        public void Clear()
        {
            var roots = children;

            foreach (var root in roots)
                DestroyHierarchy(root);

            m_Bones.Clear();
        }

        public string GetUniqueName(BoneCache bone)
        {
            Debug.Assert(Contains(bone));

            var boneName = bone.name;
            var names = m_Bones.ConvertAll(b => b.name);
            var index = IndexOf(bone);
            var count = 0;

            Debug.Assert(index < names.Count);

            for (var i = 0; i < index; ++i)
                if (names[i].Equals(boneName))
                    ++count;

            return count == 0 ? boneName : $"{boneName} ({count + 1})";
        }

        void DestroyHierarchy(TransformCache root)
        {
            Debug.Assert(root != null);

            var rootChildren = root.children;
            foreach (var child in rootChildren)
                DestroyHierarchy(child);

            skinningCache.Destroy(root);
        }
    }
}
