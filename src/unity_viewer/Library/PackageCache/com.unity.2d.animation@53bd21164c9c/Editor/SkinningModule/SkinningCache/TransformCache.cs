using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace UnityEditor.U2D.Animation
{
    internal class TransformCache : SkinningObject, IEnumerable<TransformCache>
    {
        [SerializeField]
        TransformCache m_Parent;
        [SerializeField]
        List<TransformCache> m_Children = new List<TransformCache>();
        [SerializeField]
        Vector3 m_LocalPosition;
        [SerializeField]
        Quaternion m_LocalRotation = Quaternion.identity;
        [SerializeField]
        Vector3 m_LocalScale = Vector3.one;
        [SerializeField]
        Matrix4x4 m_LocalToWorldMatrix = Matrix4x4.identity;

        public TransformCache parent => m_Parent;

        public TransformCache[] children => m_Children.ToArray();

        internal int siblingIndex
        {
            get => GetSiblingIndex();
            set => SetSiblingIndex(value);
        }

        public int childCount => m_Children.Count;

        public Vector3 localPosition
        {
            get => m_LocalPosition;
            set
            {
                m_LocalPosition = value;
                Update();
            }
        }

        public Quaternion localRotation
        {
            get => m_LocalRotation;
            set
            {
                m_LocalRotation = MathUtility.NormalizeQuaternion(value);
                Update();
            }
        }

        public Vector3 localScale
        {
            get => m_LocalScale;
            set
            {
                m_LocalScale = value;
                Update();
            }
        }

        public Vector3 position
        {
            get => parentMatrix.MultiplyPoint3x4(localPosition);
            set => localPosition = parentMatrix.inverse.MultiplyPoint3x4(value);
        }

        public Quaternion rotation
        {
            get => GetGlobalRotation();
            set => SetGlobalRotation(value);
        }

        public Vector3 right
        {
            get => localToWorldMatrix.MultiplyVector(Vector3.right).normalized;
            set => MatchDirection(Vector3.right, value);
        }

        public Vector3 up
        {
            get => localToWorldMatrix.MultiplyVector(Vector3.up).normalized;
            set => MatchDirection(Vector3.up, value);
        }

        public Vector3 forward
        {
            get => localToWorldMatrix.MultiplyVector(Vector3.forward).normalized;
            set => MatchDirection(Vector3.forward, value);
        }

        public Matrix4x4 localToWorldMatrix => m_LocalToWorldMatrix;

        public Matrix4x4 worldToLocalMatrix => localToWorldMatrix.inverse;

        Matrix4x4 parentMatrix
        {
            get
            {
                var matrix = Matrix4x4.identity;
                if (parent != null)
                    matrix = parent.localToWorldMatrix;
                return matrix;
            }
        }

        internal override void OnDestroy()
        {
            if (parent != null)
                parent.RemoveChild(this);

            m_Parent = null;
            m_Children.Clear();
        }

        void Update()
        {
            m_LocalToWorldMatrix = parentMatrix * Matrix4x4.TRS(localPosition, localRotation, localScale);

            foreach (var child in m_Children)
                child.Update();
        }

        void AddChild(TransformCache transform)
        {
            m_Children.Add(transform);
        }

        void InsertChildAt(int index, TransformCache transform)
        {
            m_Children.Insert(index, transform);
        }

        void RemoveChild(TransformCache transform)
        {
            m_Children.Remove(transform);
        }

        void RemoveChildAt(int index)
        {
            m_Children.RemoveAt(index);
        }

        int GetSiblingIndex()
        {
            if (parent == null)
                return -1;

            return parent.m_Children.IndexOf(this);
        }

        void SetSiblingIndex(int index)
        {
            if (parent == null)
                return;

            var currentIndex = parent.m_Children.IndexOf(this);
            var indexToRemove = index < currentIndex ? currentIndex + 1 : currentIndex;
            parent.InsertChildAt(index, this);
            parent.RemoveChildAt(indexToRemove);
        }

        public void SetParent(TransformCache newParent, bool worldPositionStays = true)
        {
            if (m_Parent == newParent)
                return;

            var oldPosition = position;
            var oldRotation = rotation;

            if (m_Parent != null)
                m_Parent.RemoveChild(this);

            m_Parent = newParent;

            if (m_Parent != null)
                m_Parent.AddChild(this);

            if (worldPositionStays)
            {
                position = oldPosition;
                rotation = oldRotation;
            }
            else
            {
                Update();
            }
        }

        Quaternion GetGlobalRotation()
        {
            var globalRotation = localRotation;
            var currentParent = parent;

            while (currentParent != null)
            {
                globalRotation = ScaleMulQuaternion(currentParent.localScale, globalRotation);
                globalRotation = currentParent.localRotation * globalRotation;
                currentParent = currentParent.parent;
            }

            return globalRotation;
        }

        void SetGlobalRotation(Quaternion r)
        {
            if (parent != null)
                r = parent.InverseTransformRotation(r);
            localRotation = r;
        }

        Quaternion InverseTransformRotation(Quaternion r)
        {
            if (parent != null)
                r = parent.InverseTransformRotation(r);

            r = Quaternion.Inverse(localRotation) * r;
            r = ScaleMulQuaternion(localScale, r);

            return r;
        }

        static Quaternion ScaleMulQuaternion(Vector3 scale, Quaternion q)
        {
            var s = new Vector3(ChangeSign(1f, scale.x), ChangeSign(1f, scale.y), ChangeSign(1f, scale.z));
            q.x = ChangeSign(q.x, s.y * s.z);
            q.y = ChangeSign(q.y, s.x * s.z);
            q.z = ChangeSign(q.z, s.x * s.y);
            return q;
        }

        static float ChangeSign(float x, float y)
        {
            return y < 0f ? -x : x;
        }

        void MatchDirection(Vector3 localDirection, Vector3 worldDirection)
        {
            var direction = worldToLocalMatrix.MultiplyVector(worldDirection);
            direction = Matrix4x4.TRS(Vector3.zero, localRotation, localScale).MultiplyVector(direction);
            var scaledLocalDirection = Vector3.Scale(localDirection, localScale);
            var deltaRotation = Quaternion.identity;

            if (scaledLocalDirection.sqrMagnitude > 0f)
            {
                var axis = Vector3.Cross(scaledLocalDirection, direction);
                var angle = Vector3.SignedAngle(scaledLocalDirection, direction, axis);
                deltaRotation = Quaternion.AngleAxis(angle, axis);
            }

            localRotation = deltaRotation;
        }

        IEnumerator<TransformCache> IEnumerable<TransformCache>.GetEnumerator()
        {
            return m_Children.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return (IEnumerator)m_Children.GetEnumerator();
        }
    }
}
