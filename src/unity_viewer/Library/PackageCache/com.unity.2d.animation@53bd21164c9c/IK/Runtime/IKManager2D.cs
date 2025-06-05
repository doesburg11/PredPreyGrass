using System.Collections.Generic;
using System.Linq;
using Unity.Profiling;
using UnityEngine.Scripting.APIUpdating;
using UnityEngine.U2D.Animation;
using UnityEngine.U2D.Common;

namespace UnityEngine.U2D.IK
{
    /// <summary>
    /// Component responsible for managing and updating 2D IK Solvers.
    /// </summary>
    [DefaultExecutionOrder(UpdateOrder.ikUpdateOrder)]
    [MovedFrom("UnityEngine.Experimental.U2D.IK")]
    [IconAttribute(IconUtility.IconPath + "Animation.IKManager.png")]
    [ExecuteInEditMode]
    public partial class IKManager2D : MonoBehaviour, IPreviewable
    {
#if UNITY_EDITOR
        internal static event System.Action<IKManager2D> onEnabledEditor;
        internal static event System.Action<IKManager2D> onDisabledEditor;
#endif

        [SerializeField]
        List<Solver2D> m_Solvers = new List<Solver2D>();
        [SerializeField]
        [Range(0f, 1f)]
        float m_Weight = 1f;
        [SerializeField]
        bool m_AlwaysUpdate = true;

        bool m_CullingEnabled;
        BaseCullingStrategy m_CullingStrategy;
        internal BaseCullingStrategy GetCullingStrategy() => m_CullingStrategy;

        /// <summary>
        /// Get and set the weight for solvers.
        /// </summary>
        public float weight
        {
            get => m_Weight;
            set => m_Weight = Mathf.Clamp01(value);
        }

        /// <summary>
        /// Get the Solvers that are managed by this manager.
        /// </summary>
        public List<Solver2D> solvers => m_Solvers;

        int[] m_TransformIdCache;

        /// <summary>
        /// Solvers are always updated even if the underlying Sprite Skins are not visible.
        /// </summary>
        public bool alwaysUpdate
        {
            get => m_AlwaysUpdate;
            set
            {
                m_AlwaysUpdate = value;
                ToggleCulling(!m_AlwaysUpdate);
            }
        }

        void OnEnable()
        {
            ToggleCulling(!m_AlwaysUpdate);

#if UNITY_EDITOR
            onEnabledEditor?.Invoke(this);
#endif
        }

        void OnDisable()
        {
            ToggleCulling(false);

#if UNITY_EDITOR
            onDisabledEditor?.Invoke(this);
#endif
        }

        void ToggleCulling(bool enableCulling)
        {
            if (m_CullingStrategy != null && m_CullingEnabled == enableCulling)
                return;

            m_CullingEnabled = enableCulling;
            m_CullingStrategy?.RemoveRequestingObject(this);

            if (m_CullingEnabled)
                m_CullingStrategy = CullingManager.instance.GetCullingStrategy<SpriteSkinVisibilityCullingStrategy>();
            else
                m_CullingStrategy = CullingManager.instance.GetCullingStrategy<AlwaysUpdateCullingStrategy>();

            m_CullingStrategy.AddRequestingObject(this);
        }

        void OnValidate()
        {
            m_Weight = Mathf.Clamp01(m_Weight);
            OnEditorDataValidate();
        }

        void Reset()
        {
            FindChildSolvers();
            OnEditorDataValidate();
        }

        void FindChildSolvers()
        {
            m_Solvers.Clear();

            var solvers = new List<Solver2D>();
            transform.GetComponentsInChildren<Solver2D>(true, solvers);

            foreach (var solver in solvers)
            {
                if (solver.GetComponentInParent<IKManager2D>() == this)
                    AddSolver(solver);
            }
        }

        /// <summary>
        /// Add Solver to the manager.
        /// </summary>
        /// <param name="solver">Solver to add.</param>
        public void AddSolver(Solver2D solver)
        {
            if (!m_Solvers.Contains(solver))
            {
                m_Solvers.Add(solver);
                AddSolverEditorData();
            }
        }

        /// <summary>
        /// Remove Solver from the manager.
        /// </summary>
        /// <param name="solver">Solver to remove.</param>
        public void RemoveSolver(Solver2D solver)
        {
            RemoveSolverEditorData(solver);
            m_Solvers.Remove(solver);
        }

        /// <summary>
        /// Updates the Solvers in this manager.
        /// </summary>
        public void UpdateManager()
        {
            if (m_Solvers.Count == 0)
                return;

            var profilerMarker = new ProfilerMarker("IKManager2D.UpdateManager");
            profilerMarker.Begin();

            ToggleCulling(!m_AlwaysUpdate);

            var solverInitialized = false;
            for (var i = 0; i < m_Solvers.Count; i++)
            {
                var solver = m_Solvers[i];
                if (solver == null || !solver.isActiveAndEnabled)
                    continue;

                if (!solver.isValid)
                {
                    solver.Initialize();
                    solverInitialized = true;
                }

                if (!m_CullingEnabled)
                    solver.UpdateIK(m_Weight);
            }

            if (m_CullingEnabled)
            {
                if (solverInitialized || m_TransformIdCache == null)
                    CacheSolversTransformIds();

                var canUpdate = m_AlwaysUpdate || m_CullingStrategy.AreBonesVisible(m_TransformIdCache);
                if (canUpdate)
                {
                    for (var i = 0; i < m_Solvers.Count; i++)
                    {
                        var solver = m_Solvers[i];
                        if (solver == null || !solver.isActiveAndEnabled)
                            continue;

                        solver.UpdateIK(weight);
                    }
                }
            }

            profilerMarker.End();
        }

        void CacheSolversTransformIds()
        {
            var transformCache = new HashSet<int>();
            for (var s = 0; s < solvers.Count; s++)
            {
                var solver = solvers[s];
                for (var c = 0; c < solver.chainCount; c++)
                {
                    var chain = solver.GetChain(c);
                    for (var b = 0; b < chain.transformCount; b++)
                    {
                        var boneTransform = chain.transforms[b];
                        if (boneTransform != null)
                            transformCache.Add(boneTransform.GetInstanceID());
                    }
                }
            }

            m_TransformIdCache = transformCache.ToArray();
        }

        /// <summary>
        /// Used by the animation clip preview window. Recommended to not use outside of this purpose.
        /// </summary>
        public void OnPreviewUpdate()
        {
#if UNITY_EDITOR
            if (IsInGUIUpdateLoop())
                UpdateManager();
#endif
        }

        static bool IsInGUIUpdateLoop() => Event.current != null;

        void LateUpdate()
        {
            UpdateManager();
        }

#if UNITY_EDITOR
        internal static Events.UnityEvent onDrawGizmos = new Events.UnityEvent();
        void OnDrawGizmos()
        {
            onDrawGizmos.Invoke();
        }
#endif
    }
}
