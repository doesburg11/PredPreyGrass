using System;
using System.Collections.Generic;

namespace UnityEngine.U2D.IK
{
    /// <summary>
    /// Base class used for defining culling strategies for IKManager2D.
    /// </summary>
    internal abstract class BaseCullingStrategy
    {
        public bool enabled => m_IsCullingEnabled;
        bool m_IsCullingEnabled;

        HashSet<object> m_RequestingManagers;

        /// <summary>
        /// Used to check if bone transforms should be culled.
        /// </summary>
        /// <param name="transformIds">A collection of bones' transform ids.</param>
        /// <returns>True if any bone is visible.</returns>
        public abstract bool AreBonesVisible(IList<int> transformIds);

        public void AddRequestingObject(object requestingObject)
        {
            if (!m_IsCullingEnabled)
            {
                m_IsCullingEnabled = true;

                Initialize();
            }

            m_RequestingManagers.Add(requestingObject);
        }

        public void RemoveRequestingObject(object requestingObject)
        {
            if (m_RequestingManagers.Remove(requestingObject) && m_RequestingManagers.Count == 0)
            {
                m_IsCullingEnabled = false;

                Disable();
            }
        }

        public void Initialize()
        {
            m_RequestingManagers = new HashSet<object>();
            OnInitialize();
        }

        public void Update()
        {
            OnUpdate();
        }

        public void Disable()
        {
            OnDisable();
        }

        protected virtual void OnInitialize() { }
        protected virtual void OnUpdate() { }
        protected virtual void OnDisable() { }
    }
}
