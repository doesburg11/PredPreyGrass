using UnityEngine;
using UnityEditor;
using UnityEngine.U2D;

/// <summary>
/// Inspector for SpriteShapeObjectPlacement Component.
/// </summary>
[CustomEditor(typeof(SpriteShapeObjectPlacement))]
[CanEditMultipleObjects]
public class SpriteShapeObjectPlacementEditor : Editor
{
    SerializedProperty m_SpriteShapeControllerObject;
    SerializedProperty m_StartPointObject;
    SerializedProperty m_EndPointObject;
    SerializedProperty m_SetNormalObject;
    SerializedProperty m_RatioObject;
    SerializedProperty m_ModeObject;
    static readonly GUIContent kStartPoint = new GUIContent(L10n.Tr("Start Point"));
    static readonly GUIContent kEndPoint = new GUIContent(L10n.Tr("End Point"));
    static readonly GUIContent kRatio = new GUIContent(L10n.Tr("Ratio"));

    void OnEnable()
    {
        m_SpriteShapeControllerObject = serializedObject.FindProperty("m_SpriteShapeController");
        m_StartPointObject = serializedObject.FindProperty("m_StartPoint");
        m_EndPointObject = serializedObject.FindProperty("m_EndPoint");
        m_SetNormalObject = serializedObject.FindProperty("m_SetNormal");
        m_RatioObject = serializedObject.FindProperty("m_Ratio");
        m_ModeObject = serializedObject.FindProperty("m_Mode");
    }

    /// <summary>
    /// Handle On Inspector.
    /// </summary>
    public override void OnInspectorGUI()
    {
        serializedObject.Update();
        EditorGUILayout.PropertyField(m_SpriteShapeControllerObject);
        EditorGUILayout.PropertyField(m_SetNormalObject);
        EditorGUILayout.PropertyField(m_ModeObject);
        if (m_ModeObject.enumValueIndex != 0)
        {
            EditorGUI.BeginChangeCheck();
            var startPoint = EditorGUILayout.DelayedIntField(kStartPoint, m_StartPointObject.intValue);
            var endPoint = EditorGUILayout.DelayedIntField(kEndPoint, m_EndPointObject.intValue);
            var ratio = EditorGUILayout.Slider(kRatio, m_RatioObject.floatValue, 0, 1.0f);
            if (EditorGUI.EndChangeCheck())
            {
                m_StartPointObject.intValue = startPoint;
                m_EndPointObject.intValue = endPoint;
                m_RatioObject.floatValue = ratio;
            }
        }
        serializedObject.ApplyModifiedProperties();
    }
}