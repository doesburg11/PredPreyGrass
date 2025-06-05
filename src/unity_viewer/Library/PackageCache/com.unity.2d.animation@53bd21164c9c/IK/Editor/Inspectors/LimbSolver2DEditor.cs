using UnityEngine;
using UnityEngine.U2D.IK;

namespace UnityEditor.U2D.IK
{
    /// <summary>
    /// Custom Inspector for LimbSolver2D.
    /// </summary>
    [CustomEditor(typeof(LimbSolver2D))]
    [CanEditMultipleObjects]
    internal class LimbSolver2DEditor : Solver2DEditor
    {
        static class Contents
        {
            public static readonly GUIContent effectorLabel = new GUIContent("Effector", "The last Transform of a hierarchy constrained by the target");
            public static readonly GUIContent targetLabel = new GUIContent("Target", "Transform which the effector will follow");
            public static readonly GUIContent flipLabel = new GUIContent("Flip", "Select between the two possible solutions of the solver");
        }

        SerializedProperty m_ChainProperty;
        SerializedProperty m_FlipProperty;

        void OnEnable()
        {
            m_ChainProperty = serializedObject.FindProperty("m_Chain");
            m_FlipProperty = serializedObject.FindProperty("m_Flip");
        }

        /// <summary>
        /// Custom Inspector OnInspectorGUI override.
        /// </summary>
        public override void OnInspectorGUI()
        {
            serializedObject.Update();
            EditorGUILayout.PropertyField(m_ChainProperty.FindPropertyRelative("m_EffectorTransform"), Contents.effectorLabel);
            EditorGUILayout.PropertyField(m_ChainProperty.FindPropertyRelative("m_TargetTransform"), Contents.targetLabel);
            EditorGUILayout.PropertyField(m_FlipProperty, Contents.flipLabel);

            DrawCommonSolverInspector();

            serializedObject.ApplyModifiedProperties();
        }
    }
}
