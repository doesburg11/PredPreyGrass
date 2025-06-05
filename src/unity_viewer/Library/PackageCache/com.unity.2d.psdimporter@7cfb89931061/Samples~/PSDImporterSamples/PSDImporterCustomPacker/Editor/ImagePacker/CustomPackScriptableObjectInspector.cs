using System;
using UnityEditor;
using UnityEngine;

namespace PSDImporterCustomPacker
{
    [CustomEditor(typeof(CustomPackScriptableObject))]
    class CustomPackScriptableObjectInspector : Editor
    {
        SerializedProperty m_ObjectsToApply;

        void OnEnable()
        {
            m_ObjectsToApply = serializedObject.FindProperty("m_ObjectsToApply");
        }

        public override void OnInspectorGUI()
        {
            serializedObject.Update();
            EditorGUILayout.PropertyField(m_ObjectsToApply);
            serializedObject.ApplyModifiedProperties();
            if (GUILayout.Button("Apply"))
            {
                var importer = target as CustomPackScriptableObject;
                importer?.ApplyPipeline();
            }
        }
    }
}