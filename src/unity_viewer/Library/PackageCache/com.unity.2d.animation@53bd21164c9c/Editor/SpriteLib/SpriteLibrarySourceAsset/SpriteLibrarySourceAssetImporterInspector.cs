using System;
using UnityEditor.AssetImporters;
using UnityEngine;
using UnityEngine.U2D.Animation;
using Object = UnityEngine.Object;

namespace UnityEditor.U2D.Animation
{
    [CanEditMultipleObjects]
    [CustomEditor(typeof(SpriteLibrarySourceAssetImporter))]
    internal class SpriteLibrarySourceAssetImporterInspector : ScriptedImporterEditor
    {
        static class Contents
        {
            public static readonly GUIContent openInSpriteLibraryEditor = new(L10n.Tr("Open in Sprite Library Editor"));
        }

        static class Style
        {
            public static readonly GUIContent mainAssetLabel = new("Main Library");
        }

        SerializedProperty m_PrimaryLibraryGUID;

        public override bool showImportedObject => false;
        protected override Type extraDataType => typeof(SpriteLibrarySourceAsset);

        public override void OnEnable()
        {
            base.OnEnable();
            m_PrimaryLibraryGUID = extraDataSerializedObject.FindProperty(SpriteLibrarySourceAssetPropertyString.primaryLibraryGUID);
        }

        protected override void InitializeExtraDataInstance(Object extraTarget, int targetIndex)
        {
            var assetPath = ((AssetImporter)targets[targetIndex]).assetPath;
            var savedAsset = SpriteLibrarySourceAssetImporter.LoadSpriteLibrarySourceAsset(assetPath);
            if (savedAsset != null)
            {
                // Add entries from Main Library Asset.
                if (!SpriteLibrarySourceAssetImporter.HasValidMainLibrary(savedAsset, assetPath))
                    savedAsset.SetPrimaryLibraryGUID(string.Empty);
                SpriteLibrarySourceAssetImporter.UpdateSpriteLibrarySourceAssetLibraryWithMainAsset(savedAsset);

                (extraTarget as SpriteLibrarySourceAsset).InitializeWithAsset(savedAsset);
            }
        }

        public override void OnInspectorGUI()
        {
            if (GUILayout.Button(Contents.openInSpriteLibraryEditor))
            {
                var path = ((AssetImporter)target).assetPath;
                var asset = AssetDatabase.LoadAssetAtPath<SpriteLibraryAsset>(path);
                SpriteLibraryEditor.SpriteLibraryEditorWindow.OpenWindowForAsset(asset);
            }

            GUILayout.Space(10);

            serializedObject.Update();
            extraDataSerializedObject.Update();
            DoMainAssetGUI();
            serializedObject.ApplyModifiedProperties();
            extraDataSerializedObject.ApplyModifiedProperties();

            ApplyRevertGUI();
        }

        protected override void Apply()
        {
            // Make sure that all changes are Saved / Reverted by the user if there is an instance of SpriteLibraryEditorWindow open.
            SpriteLibraryEditor.SpriteLibraryEditorWindow.HandleUnsavedChangesOnApply();
            base.Apply();

            for (var i = 0; i < targets.Length; i++)
            {
                var path = ((AssetImporter)targets[i]).assetPath;
                var sourceAsset = (SpriteLibrarySourceAsset)extraDataTargets[i];
                var savedAsset = SpriteLibrarySourceAssetImporter.LoadSpriteLibrarySourceAsset(path);
                savedAsset.InitializeWithAsset(sourceAsset);

                // Remove entries that come from Main Library Asset before saving.
                var savedLibrarySerializedObject = new SerializedObject(savedAsset);
                SpriteLibraryUtilitiesEditor.UpdateLibraryWithNewMainLibrary(null, savedLibrarySerializedObject.FindProperty(SpriteLibrarySourceAssetPropertyString.library));
                if (savedLibrarySerializedObject.hasModifiedProperties)
                    savedLibrarySerializedObject.ApplyModifiedPropertiesWithoutUndo();

                // Save asset to disk.
                SpriteLibrarySourceAssetImporter.SaveSpriteLibrarySourceAsset(savedAsset, path);
            }

            // Due to case 1418417 we can't guarantee
            // that changes will be propagated to the SpriteLibraryEditor window.
            // Until fixed, keep this line to ensure that SpriteLibraryEditor window reloads.
            SpriteLibraryEditor.SpriteLibraryEditorWindow.TriggerAssetModifiedOnApply();
        }

        void DoMainAssetGUI()
        {
            EditorGUI.BeginChangeCheck();
            if (m_PrimaryLibraryGUID.hasMultipleDifferentValues)
                EditorGUI.showMixedValue = true;
            var currentMainSpriteLibraryAsset = AssetDatabase.LoadAssetAtPath<SpriteLibraryAsset>(AssetDatabase.GUIDToAssetPath(m_PrimaryLibraryGUID.stringValue));
            var newMainLibraryAsset = EditorGUILayout.ObjectField(Style.mainAssetLabel, currentMainSpriteLibraryAsset, typeof(SpriteLibraryAsset), false) as SpriteLibraryAsset;
            if (EditorGUI.EndChangeCheck())
            {
                var successfulAssignment = true;
                for (var i = 0; i < targets.Length; ++i)
                    successfulAssignment = AssignNewMainLibrary(targets[i], extraDataTargets[i] as SpriteLibrarySourceAsset, newMainLibraryAsset);

                if (successfulAssignment)
                    m_PrimaryLibraryGUID.stringValue = AssetDatabase.AssetPathToGUID(AssetDatabase.GetAssetPath(newMainLibraryAsset));
            }

            EditorGUI.showMixedValue = false;
        }

        static bool AssignNewMainLibrary(Object target, SpriteLibrarySourceAsset extraTarget, SpriteLibraryAsset newMainLibrary)
        {
            var assetPath = ((AssetImporter)target).assetPath;
            var spriteLibraryAsset = AssetDatabase.LoadAssetAtPath<SpriteLibraryAsset>(assetPath);
            var parentChain = SpriteLibrarySourceAssetImporter.GetAssetParentChain(newMainLibrary);
            if (assetPath == AssetDatabase.GetAssetPath(newMainLibrary) || parentChain.Contains(spriteLibraryAsset))
            {
                Debug.LogWarning(TextContent.spriteLibraryCircularDependency);
                return false;
            }

            var path = ((AssetImporter)target).assetPath;
            var toSavedAsset = SpriteLibrarySourceAssetImporter.LoadSpriteLibrarySourceAsset(path);

            toSavedAsset.InitializeWithAsset(extraTarget);
            var savedLibrarySerializedObject = new SerializedObject(toSavedAsset);
            SpriteLibraryUtilitiesEditor.UpdateLibraryWithNewMainLibrary(newMainLibrary, savedLibrarySerializedObject.FindProperty(SpriteLibrarySourceAssetPropertyString.library));
            if (savedLibrarySerializedObject.hasModifiedProperties)
                savedLibrarySerializedObject.ApplyModifiedPropertiesWithoutUndo();

            return true;
        }
    }

    internal class CreateSpriteLibrarySourceAsset : ProjectWindowCallback.EndNameEditAction
    {
        const int k_SpriteLibraryAssetMenuPriority = 30;
        string m_MainLibrary;

        public override void Action(int instanceId, string pathName, string resourceFile)
        {
            var asset = CreateInstance<SpriteLibrarySourceAsset>();
            asset.SetPrimaryLibraryGUID(m_MainLibrary);

            UnityEditorInternal.InternalEditorUtility.SaveToSerializedFileAndForget(new Object[] { asset }, pathName, true);
            AssetDatabase.Refresh(ImportAssetOptions.ForceUpdate);

            m_MainLibrary = string.Empty;

            ProjectWindowUtil.ShowCreatedAsset(AssetDatabase.LoadAssetAtPath<Object>(pathName));
        }

        [MenuItem("Assets/Create/2D/Sprite Library Asset", priority = k_SpriteLibraryAssetMenuPriority)]
        static void CreateSpriteLibrarySourceAssetMenu()
        {
            var action = CreateInstance<CreateSpriteLibrarySourceAsset>();
            var icon = EditorIconUtility.LoadIconResource("Sprite Library", "Icons/Light", "Icons/Dark");
            ProjectWindowUtil.StartNameEditingIfProjectWindowExists(0, action, SpriteLibrarySourceAsset.defaultName + SpriteLibrarySourceAsset.extension, icon, null);
        }

        [MenuItem("Assets/Create/2D/Sprite Library Asset Variant", priority = k_SpriteLibraryAssetMenuPriority + 1)]
        static void CreateSpriteLibrarySourceAssetVariantMenu()
        {
            var action = CreateInstance<CreateSpriteLibrarySourceAsset>();
            var asset = SpriteLibrarySourceAssetImporter.GetAssetFromSelection();
            if (asset != null)
                action.m_MainLibrary = AssetDatabase.GUIDFromAssetPath(AssetDatabase.GetAssetPath(asset)).ToString();
            var icon = EditorIconUtility.LoadIconResource("Sprite Library", "Icons/Light", "Icons/Dark");
            ProjectWindowUtil.StartNameEditingIfProjectWindowExists(0, action, SpriteLibrarySourceAsset.defaultName + SpriteLibrarySourceAsset.extension, icon, null);
        }

        [MenuItem("Assets/Create/2D/Sprite Library Asset Variant", true, priority = k_SpriteLibraryAssetMenuPriority + 1)]
        static bool ValidateCanCreateVariant()
        {
            return SpriteLibrarySourceAssetImporter.GetAssetFromSelection() != null;
        }
    }
}
