using System;
using UnityEngine;

namespace UnityEditor.U2D.Animation
{
    [Serializable]
    internal struct Pose
    {
        public Vector3 position;
        public Quaternion rotation;
        public Matrix4x4 matrix => Matrix4x4.TRS(position, rotation, Vector3.one);

        public static Pose Create(Vector3 p, Quaternion r)
        {
            var pose = new Pose()
            {
                position = p,
                rotation = r
            };

            return pose;
        }

        public override bool Equals(object other)
        {
            return other is Pose && this == (Pose)other;
        }

        public override int GetHashCode()
        {
            return position.GetHashCode() ^ rotation.GetHashCode();
        }

        public static bool operator ==(Pose p1, Pose p2)
        {
            return p1.position == p2.position && p1.rotation == p2.rotation;
        }

        public static bool operator !=(Pose p1, Pose p2)
        {
            return !(p1 == p2);
        }
    }

    [Serializable]
    internal struct BonePose
    {
        public Pose pose;
        public float length;
        public static BonePose Create(Pose p, float l)
        {
            var pose = new BonePose()
            {
                pose = p,
                length = l
            };

            return pose;
        }

        public override bool Equals(object other)
        {
            return other is BonePose && this == (BonePose)other;
        }

        public override int GetHashCode()
        {
            return pose.GetHashCode() ^ length.GetHashCode();
        }

        public static bool operator ==(BonePose p1, BonePose p2)
        {
            return p1.pose == p2.pose && Mathf.Abs(p1.length - p2.length) < Mathf.Epsilon;
        }

        public static bool operator !=(BonePose p1, BonePose p2)
        {
            return !(p1 == p2);
        }
    }

    internal class BoneCache : TransformCache
    {
        [SerializeField]
        Color32 m_BindPoseColor;
        [SerializeField]
        Pose m_BindPose;
        [SerializeField]
        BonePose m_DefaultPose;
        [SerializeField]
        BoneCache m_ChainedChild;
        [SerializeField]
        float m_Depth;
        [SerializeField]
        float m_LocalLength = 1f;
        [SerializeField]
        bool m_IsVisible = true;
        [SerializeField]
        string m_Guid;

        public bool NotInDefaultPose()
        {
            return localPosition != m_DefaultPose.pose.position
                || localRotation != m_DefaultPose.pose.rotation
                || Mathf.Abs(localLength - m_DefaultPose.length) > Mathf.Epsilon;
        }

        public bool isVisible
        {
            get => m_IsVisible;
            set => m_IsVisible = value;
        }

        public Color bindPoseColor
        {
            get => m_BindPoseColor;
            set => m_BindPoseColor = value;
        }

        public virtual BoneCache parentBone => parent as BoneCache;

        public SkeletonCache skeleton
        {
            get
            {
                var parentSkeleton = parent as SkeletonCache;
                if (parentSkeleton != null)
                    return parentSkeleton;

                return parentBone != null ? parentBone.skeleton : null;
            }
        }

        public virtual BoneCache chainedChild
        {
            get
            {
                if (m_ChainedChild != null && m_ChainedChild.parentBone == this)
                    return m_ChainedChild;

                return null;
            }
            set
            {
                if (m_ChainedChild != value)
                {
                    if (value == null || value.parentBone == this)
                    {
                        m_ChainedChild = value;
                        if (m_ChainedChild != null)
                            OrientToChainedChild(false);
                    }
                }
            }
        }

        Vector3 localEndPosition => Vector3.right * localLength;

        public Vector3 endPosition
        {
            get => localToWorldMatrix.MultiplyPoint3x4(localEndPosition);
            set
            {
                if (chainedChild != null)
                    return;

                var direction = value - position;
                right = direction;
                length = direction.magnitude;
            }
        }

        public BonePose localPose
        {
            get => BonePose.Create(Pose.Create(localPosition, localRotation), localLength);
            set
            {
                localPosition = value.pose.position;
                localRotation = value.pose.rotation;
                localLength = value.length;
            }
        }

        public BonePose worldPose
        {
            get => BonePose.Create(Pose.Create(position, rotation), length);
            set
            {
                position = value.pose.position;
                rotation = value.pose.rotation;
                length = value.length;
            }
        }

        public Pose bindPose => m_BindPose;

        public string guid
        {
            get => m_Guid;
            set => m_Guid = value;
        }

        public float depth
        {
            get => m_Depth;
            set => m_Depth = value;
        }
        public float localLength
        {
            get => m_LocalLength;
            set => m_LocalLength = Mathf.Max(0f, value);
        }

        public float length
        {
            get => localToWorldMatrix.MultiplyVector(localEndPosition).magnitude;
            set => m_LocalLength = worldToLocalMatrix.MultiplyVector(right * Mathf.Max(0f, value)).magnitude;
        }

        internal Pose[] GetChildrenWoldPose()
        {
            return Array.ConvertAll(children, c => Pose.Create(c.position, c.rotation));
        }

        internal void SetChildrenWorldPose(Pose[] worldPoses)
        {
            var childrenArray = children;

            Debug.Assert(childrenArray.Length == worldPoses.Length);

            for (var i = 0; i < childrenArray.Length; ++i)
            {
                var child = childrenArray[i];
                var pose = worldPoses[i];

                child.position = pose.position;
                child.rotation = pose.rotation;
            }
        }

        internal override void OnDestroy()
        {
            base.OnDestroy();
            m_ChainedChild = null;
        }

        public new void SetParent(TransformCache newParent, bool worldPositionStays = true)
        {
            if (parentBone != null && parentBone.chainedChild == this)
                parentBone.chainedChild = null;

            base.SetParent(newParent, worldPositionStays);

            if (parentBone != null && parentBone.chainedChild == null && (parentBone.endPosition - position).sqrMagnitude < 0.001f)
                parentBone.chainedChild = this;
        }

        public void OrientToChainedChild(bool freezeChildren)
        {
            Debug.Assert(chainedChild != null);

            var childPosition = chainedChild.position;
            var childRotation = chainedChild.rotation;

            Pose[] childrenWorldPose = null;

            if (freezeChildren)
                childrenWorldPose = GetChildrenWoldPose();

            right = childPosition - position;

            if (freezeChildren)
            {
                SetChildrenWorldPose(childrenWorldPose);
            }
            else
            {
                chainedChild.position = childPosition;
                chainedChild.rotation = childRotation;
            }

            length = (childPosition - position).magnitude;
        }

        public void SetDefaultPose()
        {
            m_DefaultPose = localPose;

            if (IsUnscaled())
                m_BindPose = worldPose.pose;
            else
                throw new Exception("BindPose cannot be set under global scale");
        }

        public void RestoreDefaultPose()
        {
            localPose = m_DefaultPose;
        }

        bool IsUnscaled()
        {
            var currentTransform = this as TransformCache;

            while (currentTransform != null)
            {
                var scale = currentTransform.localScale;
                var isUnscaled = Mathf.Approximately(scale.x, 1f) && Mathf.Approximately(scale.y, 1f) && Mathf.Approximately(scale.z, 1f);

                if (!isUnscaled)
                    return false;

                currentTransform = currentTransform.parent;
            }

            return true;
        }
    }
}
