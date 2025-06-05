using System;
using UnityEngine;

namespace UnityEditor.U2D.Common.Path
{
    internal class ControlPointSettings
    {
        const string kControlPointRKey = UserSettings.kSettingsUniqueKey + "PathEditorSetting.ControlPointR";
        const string kControlPointGKey = UserSettings.kSettingsUniqueKey + "PathEditorSetting.ControlPointG";
        const string kControlPointBKey = UserSettings.kSettingsUniqueKey + "PathEditorSetting.ControlPointB";
        const string kControlPointAKey = UserSettings.kSettingsUniqueKey + "PathEditorSetting.ControlPointA";
        static readonly GUIContent kControlPointKeyContent = EditorGUIUtility.TrTextContent("ControlPoint Color");

        const string kControlPointSRKey = UserSettings.kSettingsUniqueKey + "PathEditorSetting.ControlPointSR";
        const string kControlPointSGKey = UserSettings.kSettingsUniqueKey + "PathEditorSetting.ControlPointSG";
        const string kControlPointSBKey = UserSettings.kSettingsUniqueKey + "PathEditorSetting.ControlPointSB";
        const string kControlPointSAKey = UserSettings.kSettingsUniqueKey + "PathEditorSetting.ControlPointSA";
        static readonly GUIContent kControlPointSKeyContent = EditorGUIUtility.TrTextContent("ControlPoint (Selected)");      
        
        static bool showHandle = true;
        
        public static Color controlPointColor
        {
            get
            {
                return new Color()
                {
                    r = EditorPrefs.GetFloat(kControlPointRKey, 1),
                    g = EditorPrefs.GetFloat(kControlPointGKey, 1),
                    b = EditorPrefs.GetFloat(kControlPointBKey, 1),
                    a = EditorPrefs.GetFloat(kControlPointAKey, 1)
                };
            }
            set
            {
                EditorPrefs.SetFloat(kControlPointRKey, value.r);
                EditorPrefs.SetFloat(kControlPointGKey, value.g);
                EditorPrefs.SetFloat(kControlPointBKey, value.b);
                EditorPrefs.SetFloat(kControlPointAKey, value.a);
            }
        }

        public static Color controlPointSelectedColor
        {
            get
            {
                return new Color()
                {
                    r = EditorPrefs.GetFloat(kControlPointSRKey, 1),
                    g = EditorPrefs.GetFloat(kControlPointSGKey, 235.0f / 255.0f),
                    b = EditorPrefs.GetFloat(kControlPointSBKey, 4F / 255F),
                    a = EditorPrefs.GetFloat(kControlPointSAKey, 1)
                };
            }
            set
            {
                EditorPrefs.SetFloat(kControlPointSRKey, value.r);
                EditorPrefs.SetFloat(kControlPointSGKey, value.g);
                EditorPrefs.SetFloat(kControlPointSBKey, value.b);
                EditorPrefs.SetFloat(kControlPointSAKey, value.a);
            }
        }

        internal void SetDefault()
        {
            controlPointColor = Color.white;
            controlPointSelectedColor = new Color(1.0f, 235.0f / 255.0f, 4.0f / 255.0f, 1.0f);
        }

        public void OnGUI()
        {
            EditorGUILayout.Space(8);
            showHandle = EditorGUILayout.BeginFoldoutHeaderGroup(showHandle, "Control Points");
            if (showHandle)
            {
                EditorGUI.indentLevel++;

                EditorGUI.BeginChangeCheck();
                var sc = EditorGUILayout.ColorField(kControlPointKeyContent, controlPointColor);
                if (EditorGUI.EndChangeCheck())
                    controlPointColor = sc;
                
                EditorGUI.BeginChangeCheck();
                var sh = EditorGUILayout.ColorField(kControlPointSKeyContent, controlPointSelectedColor);
                if (EditorGUI.EndChangeCheck())
                    controlPointSelectedColor = sh;

                EditorGUI.indentLevel--;
            }
            EditorGUILayout.EndFoldoutHeaderGroup();
        }
    }
    internal class HandleSettings
    {
        const string kSplineRKey = UserSettings.kSettingsUniqueKey + "PathEditorSetting.SplineR";
        const string kSplineGKey = UserSettings.kSettingsUniqueKey + "PathEditorSetting.SplineG";
        const string kSplineBKey = UserSettings.kSettingsUniqueKey + "PathEditorSetting.SplineB";
        const string kSplineAKey = UserSettings.kSettingsUniqueKey + "PathEditorSetting.SplineA";
        static readonly GUIContent kSplineKeyContent = EditorGUIUtility.TrTextContent("Spline Color");

        const string kSplineHRKey = UserSettings.kSettingsUniqueKey + "PathEditorSetting.SplineHR";
        const string kSplineHGKey = UserSettings.kSettingsUniqueKey + "PathEditorSetting.SplineHG";
        const string kSplineHBKey = UserSettings.kSettingsUniqueKey + "PathEditorSetting.SplineHB";
        const string kSplineHAKey = UserSettings.kSettingsUniqueKey + "PathEditorSetting.SplineHA";
        static readonly GUIContent kSplineHKeyContent = EditorGUIUtility.TrTextContent("Spline Color (Hovered)");           
        
        const string kTangentRKey = UserSettings.kSettingsUniqueKey + "PathEditorSetting.TangentR";
        const string kTangentGKey = UserSettings.kSettingsUniqueKey + "PathEditorSetting.TangentG";
        const string kTangentBKey = UserSettings.kSettingsUniqueKey + "PathEditorSetting.TangentB";
        const string kTangentAKey = UserSettings.kSettingsUniqueKey + "PathEditorSetting.TangentA";
        static readonly GUIContent kTangentKeyContent = EditorGUIUtility.TrTextContent("Tangent Color");        
        
        static bool showHandle = true;
        
        public static Color splineColor
        {
            get
            {
                return new Color()
                {
                    r = EditorPrefs.GetFloat(kSplineRKey, 1),
                    g = EditorPrefs.GetFloat(kSplineGKey, 1),
                    b = EditorPrefs.GetFloat(kSplineBKey, 1),
                    a = EditorPrefs.GetFloat(kSplineAKey, 1)
                };
            }
            set
            {
                EditorPrefs.SetFloat(kSplineRKey, value.r);
                EditorPrefs.SetFloat(kSplineGKey, value.g);
                EditorPrefs.SetFloat(kSplineBKey, value.b);
                EditorPrefs.SetFloat(kSplineAKey, value.a);
            }
        }

        public static Color splineHoveredColor
        {
            get
            {
                return new Color()
                {
                    r = EditorPrefs.GetFloat(kSplineHRKey, 1),
                    g = EditorPrefs.GetFloat(kSplineHGKey, 235.0f / 255.0f),
                    b = EditorPrefs.GetFloat(kSplineHBKey, 4F / 255F),
                    a = EditorPrefs.GetFloat(kSplineHAKey, 1)
                };
            }
            set
            {
                EditorPrefs.SetFloat(kSplineHRKey, value.r);
                EditorPrefs.SetFloat(kSplineHGKey, value.g);
                EditorPrefs.SetFloat(kSplineHBKey, value.b);
                EditorPrefs.SetFloat(kSplineHAKey, value.a);
            }
        }

        public static Color tangentColor
        {
            get
            {
                return new Color()
                {
                    r = EditorPrefs.GetFloat(kTangentRKey, 1),
                    g = EditorPrefs.GetFloat(kTangentGKey, 235.0f / 255.0f),
                    b = EditorPrefs.GetFloat(kTangentBKey, 4F / 255F),
                    a = EditorPrefs.GetFloat(kTangentAKey, 1)
                };
            }
            set
            {
                EditorPrefs.SetFloat(kTangentRKey, value.r);
                EditorPrefs.SetFloat(kTangentGKey, value.g);
                EditorPrefs.SetFloat(kTangentBKey, value.b);
                EditorPrefs.SetFloat(kTangentAKey, value.a);
            }
        }
        
        internal void SetDefault()
        {
            splineColor = Color.white;
            splineHoveredColor = new Color(1.0f, 235.0f / 255.0f, 4.0f / 255.0f, 1.0f);
            tangentColor = new Color(1.0f, 235.0f / 255.0f, 4.0f / 255.0f, 1.0f);
        }        

        public void OnGUI()
        {
            EditorGUILayout.Space(8);
            showHandle = EditorGUILayout.BeginFoldoutHeaderGroup(showHandle, "Splines and Tangent");
            if (showHandle)
            {
                EditorGUI.indentLevel++;

                EditorGUI.BeginChangeCheck();
                var sc = EditorGUILayout.ColorField(kSplineKeyContent, splineColor);
                if (EditorGUI.EndChangeCheck())
                    splineColor = sc;
                
                EditorGUI.BeginChangeCheck();
                var sh = EditorGUILayout.ColorField(kSplineHKeyContent, splineHoveredColor);
                if (EditorGUI.EndChangeCheck())
                    splineHoveredColor = sh;
                
                EditorGUI.BeginChangeCheck();
                var tc = EditorGUILayout.ColorField(kTangentKeyContent, tangentColor);
                if (EditorGUI.EndChangeCheck())
                    tangentColor = tc;                
                
                EditorGUI.indentLevel--;
            }
            EditorGUILayout.EndFoldoutHeaderGroup();
        }
    }    
    
    internal class UserSettings : SettingsProvider
    {
        public const string kSettingsUniqueKey = "UnityEditor.U2D.SpriteShape/";
        private static ControlPointSettings s_ControlPointSettings = new ControlPointSettings();
        private static HandleSettings s_HandleSettings = new HandleSettings();
        
        public UserSettings() : base("Preferences/2D/SpriteShape", SettingsScope.User)
        {
            guiHandler = OnGUI;
        }

        [SettingsProvider]
        private static SettingsProvider CreateSettingsProvider()
        {
            return new UserSettings()
            {
                guiHandler = SettingsGUI
            };
        }

        private static void SettingsGUI(string searchContext)
        {
            s_ControlPointSettings.OnGUI();
            s_HandleSettings.OnGUI();
            EditorGUILayout.Space();

            if (GUILayout.Button("Use Defaults", new GUILayoutOption[] { GUILayout.Width(100)}))
            {
                s_ControlPointSettings.SetDefault();
                s_HandleSettings.SetDefault();
            }
        }
    }
}
