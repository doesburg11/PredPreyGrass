using System.Collections.Generic;
using System.Linq;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.U2D.Common;
using UnityEngine.U2D.IK;
using UnityEngine.U2D.Animation;

namespace UnityEditor.U2D.IK
{
    [DefaultExecutionOrder(UpdateOrder.ikUpdateOrder - 1)]
    internal class IKEditorManager : ScriptableObject
    {
        static IKEditorManager s_Instance;

        readonly HashSet<IKManager2D> m_DirtyManagers = new HashSet<IKManager2D>();
        readonly HashSet<IKManager2D> m_IKManagers = new HashSet<IKManager2D>();
        readonly Dictionary<IKChain2D, Vector3> m_ChainPositionOverrides = new Dictionary<IKChain2D, Vector3>();
        readonly List<Vector3> m_TargetPositions = new List<Vector3>();

        GameObject m_Helper;
        GameObject[] m_SelectedGameobjects;

        internal bool isDraggingATool { get; private set; }

        bool m_Initialized;

        [InitializeOnLoadMethod]
        static void CreateInstance()
        {
            if (s_Instance != null)
                return;

            var ikManagers = Resources.FindObjectsOfTypeAll<IKEditorManager>();
            if (ikManagers.Length > 0)
                s_Instance = ikManagers[0];
            else
                s_Instance = ScriptableObject.CreateInstance<IKEditorManager>();
            s_Instance.hideFlags = HideFlags.HideAndDontSave;
        }

        public static IKEditorManager instance
        {
            get
            {
                if (s_Instance == null)
                    CreateInstance();
                return s_Instance;
            }
        }

        void OnEnable()
        {
            if (s_Instance == null)
                s_Instance = this;

            IKManager2D.onEnabledEditor += AddIKManager2D;
            IKManager2D.onDisabledEditor += RemoveIKManager2D;
        }

        void OnDisable()
        {
            IKManager2D.onEnabledEditor -= AddIKManager2D;
            IKManager2D.onDisabledEditor -= RemoveIKManager2D;

            Dispose();
        }

        public void Initialize()
        {
            var currentStage = StageUtility.GetCurrentStageHandle();
            var managers = currentStage.FindComponentsOfType<IKManager2D>().Where(x => x.gameObject.scene.isLoaded).ToArray();
            foreach (var ikManager2D in managers)
                m_IKManagers.Add(ikManager2D);

            RegisterCallbacks();

            m_Initialized = true;
        }

        void Dispose()
        {
            UnregisterCallbacks();
            Clear();

            m_Initialized = false;
        }

        void AddIKManager2D(IKManager2D manager)
        {
            if (manager == null)
                return;

            if (!m_Initialized)
                Initialize();

            m_IKManagers.Add(manager);
        }

        void RemoveIKManager2D(IKManager2D manager)
        {
            m_IKManagers.Remove(manager);

            if (m_IKManagers.Count == 0)
                Dispose();
        }

        private void RegisterCallbacks()
        {
#if UNITY_2019_1_OR_NEWER
            SceneView.duringSceneGui += OnSceneGUI;
#else
            SceneView.onSceneGUIDelegate += OnSceneGUI;
#endif
            Selection.selectionChanged += OnSelectionChanged;
        }

        private void UnregisterCallbacks()
        {
#if UNITY_2019_1_OR_NEWER
            SceneView.duringSceneGui -= OnSceneGUI;
#else
            SceneView.onSceneGUIDelegate -= OnSceneGUI;
#endif
            Selection.selectionChanged -= OnSelectionChanged;
        }

        private bool m_EnableGizmos;
        private bool m_CurrentEnableGizmoState;

        void OnDrawGizmos()
        {
            m_EnableGizmos = true;
            IKManager2D.onDrawGizmos.RemoveListener(OnDrawGizmos);
        }

        public void CheckGizmoToggle()
        {
            //Ignore events other than Repaint
            if (Event.current.type != EventType.Repaint)
                return;

            if (m_CurrentEnableGizmoState != m_EnableGizmos)
                SceneView.RepaintAll();

            m_CurrentEnableGizmoState = m_EnableGizmos;

            //Assume the Gizmo toggle is disabled and listen to the event again
            m_EnableGizmos = false;
            IKManager2D.onDrawGizmos.RemoveListener(OnDrawGizmos);
            IKManager2D.onDrawGizmos.AddListener(OnDrawGizmos);
        }

        private void OnSelectionChanged()
        {
            m_SelectedGameobjects = null;
        }

        void Clear()
        {
            m_IKManagers.Clear();
            m_DirtyManagers.Clear();
            m_ChainPositionOverrides.Clear();
        }

        public IKManager2D FindManager(Solver2D solver)
        {
            foreach (IKManager2D manager in m_IKManagers)
            {
                if (manager == null)
                    continue;

                foreach (Solver2D s in manager.solvers)
                {
                    if (s == null)
                        continue;

                    if (s == solver)
                        return manager;
                }
            }

            return null;
        }

        public void Record(Solver2D solver, string undoName)
        {
            var manager = FindManager(solver);

            DoUndo(manager, undoName, true);
        }

        public void RegisterUndo(Solver2D solver, string undoName)
        {
            var manager = FindManager(solver);

            DoUndo(manager, undoName, false);
        }

        public void Record(IKManager2D manager, string undoName)
        {
            DoUndo(manager, undoName, true);
        }

        public void RegisterUndo(IKManager2D manager, string undoName)
        {
            DoUndo(manager, undoName, false);
        }

        private void DoUndo(IKManager2D manager, string undoName, bool record)
        {
            if (manager == null)
                return;
            foreach (var solver in manager.solvers)
            {
                if (solver == null || !solver.isActiveAndEnabled)
                    continue;

                if (!solver.isValid)
                    solver.Initialize();

                if (!solver.isValid)
                    continue;

                for (int i = 0; i < solver.chainCount; ++i)
                {
                    var chain = solver.GetChain(i);

                    if (record)
                    {
                        foreach (var t in chain.transforms)
                            Undo.RecordObject(t, undoName);

                        if (chain.target)
                            Undo.RecordObject(chain.target, undoName);
                    }
                    else
                    {
                        foreach (var t in chain.transforms)
                            Undo.RegisterCompleteObjectUndo(t, undoName);

                        if (chain.target)
                            Undo.RegisterCompleteObjectUndo(chain.target, undoName);
                    }
                }
            }
        }

        public void UpdateManagerImmediate(IKManager2D manager, bool recordRootLoops)
        {
            SetManagerDirty(manager);
            UpdateDirtyManagers(recordRootLoops);
        }

        public void UpdateSolverImmediate(Solver2D solver, bool recordRootLoops)
        {
            SetSolverDirty(solver);
            UpdateDirtyManagers(recordRootLoops);
        }

        public void SetChainPositionOverride(IKChain2D chain, Vector3 position)
        {
            m_ChainPositionOverrides[chain] = position;
        }

        private bool IsViewToolActive()
        {
            int button = Event.current.button;
            return Tools.current == Tool.View || Event.current.alt || (button == 1) || (button == 2);
        }

        private bool IsDraggingATool()
        {
            //If a tool has used EventType.MouseDrag, we won't be able to detect it. Instead we check for delta magnitude
            return GUIUtility.hotControl != 0 && Event.current.button == 0 && Event.current.delta.sqrMagnitude > 0f && !IsViewToolActive();
        }

        private void OnSceneGUI(SceneView sceneView)
        {
            CheckGizmoToggle();
            if (!m_CurrentEnableGizmoState)
                return;

            if (m_SelectedGameobjects == null)
                m_SelectedGameobjects = Selection.gameObjects;

            foreach (var ikManager2D in m_IKManagers)
            {
                if (ikManager2D != null && ikManager2D.isActiveAndEnabled)
                {
                    if (!EditorSceneManager.IsPreviewSceneObject(ikManager2D))
                    {
                        if (PrefabStageUtility.GetCurrentPrefabStage() == null)
                            IKGizmos.instance.DoSolversGUI(ikManager2D);
                    }
                    else
                    {
                        if (PrefabStageUtility.GetCurrentPrefabStage()?.scene == ikManager2D.gameObject.scene)
                            IKGizmos.instance.DoSolversGUI(ikManager2D);
                    }
                }
            }

            if (!IKGizmos.instance.isDragging && IsDraggingATool())
            {
                //We expect the object to be selected while dragged
                foreach (var gameObject in m_SelectedGameobjects)
                {
                    if (gameObject != null && gameObject.transform != null)
                        SetDirtySolversAffectedByTransform(gameObject.transform);
                }

                if (m_DirtyManagers.Count > 0 && !isDraggingATool)
                {
                    isDraggingATool = true;
                    Undo.SetCurrentGroupName("IK Update");

                    RegisterUndoForDirtyManagers();
                }
            }

            if (GUIUtility.hotControl == 0)
                isDraggingATool = false;
        }

        private void SetSolverDirty(Solver2D solver)
        {
            if (solver && solver.isValid && solver.isActiveAndEnabled)
                SetManagerDirty(FindManager(solver));
        }

        private void SetManagerDirty(IKManager2D manager)
        {
            if (manager && manager.isActiveAndEnabled)
                m_DirtyManagers.Add(manager);
        }

        private void SetDirtySolversAffectedByTransform(Transform transform)
        {
            foreach (var manager in m_IKManagers)
            {
                if (manager != null && manager.isActiveAndEnabled)
                {
                    var dirty = false;
                    var solvers = manager.solvers;
                    for (var s = 0; s < solvers.Count; s++)
                    {
                        if (dirty)
                            break;

                        var solver = solvers[s];
                        if (solver != null && solver.isValid)
                        {
                            for (var c = 0; c < solver.chainCount; ++c)
                            {
                                var chain = solver.GetChain(c);

                                if (chain.target == null)
                                    continue;

                                if (!(IKUtility.IsDescendentOf(chain.target, transform) && IKUtility.IsDescendentOf(chain.rootTransform, transform)) &&
                                    (chain.target == transform || IKUtility.IsDescendentOf(chain.target, transform) || IKUtility.IsDescendentOf(chain.effector, transform)))
                                {
                                    SetManagerDirty(manager);
                                    dirty = true;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }

        private void RegisterUndoForDirtyManagers()
        {
            foreach (var manager in m_DirtyManagers)
                RegisterUndo(manager, Undo.GetCurrentGroupName());
        }

        private void UpdateDirtyManagers(bool recordRootLoops)
        {
            foreach (var manager in m_DirtyManagers)
            {
                if (manager == null || !manager.isActiveAndEnabled)
                    continue;

                foreach (var solver in manager.solvers)
                {
                    if (solver == null || !solver.isActiveAndEnabled)
                        continue;

                    if (!solver.isValid)
                        solver.Initialize();

                    if (!solver.isValid)
                        continue;

                    if (solver.allChainsHaveTargets)
                        solver.UpdateIK(manager.weight);
                    else if (PrepareTargetOverrides(solver))
                        solver.UpdateIK(m_TargetPositions, manager.weight);

                    for (int i = 0; i < solver.chainCount; ++i)
                    {
                        var chain = solver.GetChain(i);

                        if (recordRootLoops)
                            InternalEngineBridge.SetLocalEulerHint(chain.rootTransform);

                        if (solver.constrainRotation && chain.target != null)
                            InternalEngineBridge.SetLocalEulerHint(chain.effector);
                    }
                }
            }

            m_DirtyManagers.Clear();
            m_ChainPositionOverrides.Clear();
        }

        private bool PrepareTargetOverrides(Solver2D solver)
        {
            m_TargetPositions.Clear();

            for (int i = 0; i < solver.chainCount; ++i)
            {
                var chain = solver.GetChain(i);

                Vector3 positionOverride;
                if (!m_ChainPositionOverrides.TryGetValue(chain, out positionOverride))
                {
                    m_TargetPositions.Clear();
                    return false;
                }

                m_TargetPositions.Add(positionOverride);
            }

            return true;
        }
    }
}
