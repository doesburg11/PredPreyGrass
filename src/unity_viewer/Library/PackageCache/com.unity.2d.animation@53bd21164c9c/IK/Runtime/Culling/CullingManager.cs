using System;
using System.Collections.Generic;
using Unity.Profiling;

namespace UnityEngine.U2D.IK
{
    /// <summary>
    /// Class used for checking visibility of SpriteSkins' bones.
    /// </summary>
    [ExecuteInEditMode]
    internal class CullingManager : MonoBehaviour
    {
        static CullingManager s_Instance;

        public static CullingManager instance
        {
            get
            {
                if (s_Instance == null)
                {
                    var managers = FindObjectsByType<CullingManager>(FindObjectsSortMode.None);
                    s_Instance = managers.Length > 0 ? managers[0] : CreateNewManager();
                    s_Instance.Initialize();
                }

                return s_Instance;
            }
        }

        static CullingManager CreateNewManager()
        {
            var newGameObject = new GameObject("Culling Manager")
            {
                hideFlags = HideFlags.HideAndDontSave
            };
#if !UNITY_EDITOR
            GameObject.DontDestroyOnLoad(newGameObject);
#endif

            var cullingManager = newGameObject.AddComponent<CullingManager>();
            return cullingManager;
        }

        ProfilerMarker m_ProfilerMarker = new($"{nameof(CullingManager)}.{nameof(OnUpdate)}");
        Dictionary<Type, BaseCullingStrategy> m_CullingStrategies;

        void Initialize()
        {
            m_CullingStrategies = new Dictionary<Type, BaseCullingStrategy>();

            AddCullingStrategy(new AlwaysUpdateCullingStrategy());
            AddCullingStrategy(new SpriteSkinVisibilityCullingStrategy());
        }

        void Update()
        {
            OnUpdate();
        }

        void OnUpdate()
        {
            m_ProfilerMarker.Begin();

            if (m_CullingStrategies != null)
            {
                foreach (var cullingStrategy in m_CullingStrategies.Values)
                {
                    if (cullingStrategy.enabled)
                        cullingStrategy.Update();
                }
            }

            m_ProfilerMarker.End();
        }

        public void AddCullingStrategy(BaseCullingStrategy newCullingStrategy)
        {
            var strategyType = newCullingStrategy.GetType();
            if (m_CullingStrategies.ContainsKey(strategyType))
                return;

            m_CullingStrategies[newCullingStrategy.GetType()] = newCullingStrategy;
        }

        public void RemoveCullingStrategy(BaseCullingStrategy strategyToRemove)
        {
            var strategyType = strategyToRemove.GetType();
            if (!m_CullingStrategies.ContainsKey(strategyType))
                return;

            var cullingStrategy = m_CullingStrategies[strategyType];
            if (cullingStrategy == strategyToRemove)
                m_CullingStrategies.Remove(strategyType);
        }

        public T GetCullingStrategy<T>() where T : BaseCullingStrategy
        {
            var requestedType = typeof(T);
            if (!m_CullingStrategies.ContainsKey(requestedType))
                return null;

            return (T)m_CullingStrategies[requestedType];
        }
    }
}
