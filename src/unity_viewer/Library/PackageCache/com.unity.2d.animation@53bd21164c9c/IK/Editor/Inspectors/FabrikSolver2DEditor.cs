using UnityEngine;
using UnityEngine.U2D.IK;

namespace UnityEditor.U2D.IK
{
    /// <summary>
    /// Custom Inspector for FabrikSolver2D.
    /// </summary>
    [CustomEditor(typeof(FabrikSolver2D))]
    [CanEditMultipleObjects]
    internal class FabrikSolver2DEditor : Solver2DEditor
    {
        static class Contents
        {
            public static readonly GUIContent effectorLabel = new GUIContent("Effector", "The last Transform of a hierarchy constrained by the target");
            public static readonly GUIContent targetLabel = new GUIContent("Target", "Transform which the effector will follow");
            public static readonly GUIContent chainLengthLabel = new GUIContent("Chain Length", "Number of Transforms handled by the IK");
            public static readonly GUIContent iterationsLabel = new GUIContent("Iterations", "Number of iterations the IK solver is run per frame");
            public static readonly GUIContent toleranceLabel = new GUIContent("Tolerance", "How close the target is to the goal to be considered as successful");
        }

        SerializedProperty m_TargetProperty;
        SerializedProperty m_EffectorProperty;
        SerializedProperty m_TransformCountProperty;
        SerializedProperty m_IterationsProperty;
        SerializedProperty m_ToleranceProperty;
        FabrikSolver2D m_Solver;

        void OnEnable()
        {
            m_Solver = target as FabrikSolver2D;
            var chainProperty = serializedObject.FindProperty("m_Chain");
            m_TargetProperty = chainProperty.FindPropertyRelative("m_TargetTransform");
            m_EffectorProperty = chainProperty.FindPropertyRelative("m_EffectorTransform");
            m_TransformCountProperty = chainProperty.FindPropertyRelative("m_TransformCount");
            m_IterationsProperty = serializedObject.FindProperty("m_Iterations");
            m_ToleranceProperty = serializedObject.FindProperty("m_Tolerance");
        }

        /// <summary>
        /// Custom Inspector OnInspectorGUI override.
        /// </summary>
        public override void OnInspectorGUI()
        {
            var chain = m_Solver.GetChain(0);

            serializedObject.Update();
            EditorGUILayout.PropertyField(m_EffectorProperty, Contents.effectorLabel);
            EditorGUILayout.PropertyField(m_TargetProperty, Contents.targetLabel);
            EditorGUILayout.IntSlider(m_TransformCountProperty, 0, IKUtility.GetMaxChainCount(chain), Contents.chainLengthLabel);
            EditorGUILayout.PropertyField(m_IterationsProperty, Contents.iterationsLabel);
            EditorGUILayout.PropertyField(m_ToleranceProperty, Contents.toleranceLabel);

            DrawCommonSolverInspector();

            serializedObject.ApplyModifiedProperties();
        }
    }
}
