using System.Collections.Generic;
using UnityEngine;
using UnityEditorInternal;
using UnityEngine.U2D.Animation;
using UnityEngine.U2D;

namespace UnityEditor.U2D.Animation
{
    [CustomEditor(typeof(SpriteSkin))]
    [CanEditMultipleObjects]
    class SpriteSkinEditor : Editor
    {
        static class Contents
        {
            public static readonly GUIContent listHeaderLabel = new GUIContent("Bones", "GameObject Transform to represent the Bones defined by the Sprite that is currently used for deformation.");
            public static readonly GUIContent rootBoneLabel = new GUIContent("Root Bone", "GameObject Transform to represent the Root Bone.");
            public static readonly string spriteNotFound = L10n.Tr("Sprite not found in SpriteRenderer");
            public static readonly string spriteHasNoSkinningInformation = L10n.Tr("Sprite has no Bind Poses");
            public static readonly string spriteHasNoWeights = L10n.Tr("Sprite has no weights");
            public static readonly string rootTransformNotFound = L10n.Tr("Root Bone not set");
            public static readonly string invalidTransformArray = L10n.Tr("Bone list is invalid");
            public static readonly string transformArrayContainsNull = L10n.Tr("Bone list contains unassigned references");
            public static readonly string invalidTransformArrayLength = L10n.Tr("The number of Sprite's Bind Poses and the number of Transforms should match");
            public static readonly string invalidBoneWeights = L10n.Tr("Bone weights are invalid");
            public static readonly GUIContent alwaysUpdate = new GUIContent("Always Update", "Executes deformation of SpriteSkin even when the associated SpriteRenderer has been culled and is not visible.");
            public static readonly GUIContent autoRebind = new GUIContent("Auto Rebind", "When the Sprite in SpriteRenderer is changed, SpriteSkin will try to look for the Transforms that is needed for the Sprite using the Root Bone Tranform as parent.");
        }

        SerializedProperty m_RootBoneProperty;
        SerializedProperty m_BoneTransformsProperty;
        SerializedProperty m_AlwaysUpdateProperty;
        SerializedProperty m_AutoRebindProperty;

        SpriteSkin[] m_SpriteSkins;
        Sprite[] m_CurrentSprites;
        ReorderableList m_ReorderableList;

        bool m_NeedsRebind = false;
        bool m_BoneFold = true;

        void OnEnable()
        {
            var listOfSkins = new List<SpriteSkin>(targets.Length);
            foreach (var obj in targets)
            {
                if (obj is SpriteSkin skin)
                {
                    listOfSkins.Add(skin);
                    skin.OnEditorEnable();
                }
            }

            m_SpriteSkins = listOfSkins.ToArray();

            m_RootBoneProperty = serializedObject.FindProperty("m_RootBone");

            m_BoneTransformsProperty = serializedObject.FindProperty("m_BoneTransforms");
            m_AlwaysUpdateProperty = serializedObject.FindProperty("m_AlwaysUpdate");
            m_AutoRebindProperty = serializedObject.FindProperty("m_AutoRebind");

            m_CurrentSprites = new Sprite[m_SpriteSkins.Length];

            UpdateSpriteCache();
            SetupReorderableList();

            Undo.undoRedoPerformed += UndoRedoPerformed;
        }

        void OnDestroy()
        {
            Undo.undoRedoPerformed -= UndoRedoPerformed;
        }

        void UndoRedoPerformed()
        {
            UpdateSpriteCache();
        }

        void UpdateSpriteCache()
        {
            for (var i = 0; i < m_SpriteSkins.Length; ++i)
            {
                m_CurrentSprites[i] = m_SpriteSkins[i].sprite;
            }
        }

        public override void OnInspectorGUI()
        {
            serializedObject.Update();

            EditorGUILayout.PropertyField(m_AlwaysUpdateProperty, Contents.alwaysUpdate);

            EditorGUI.BeginChangeCheck();
            EditorGUILayout.PropertyField(m_AutoRebindProperty, Contents.autoRebind);
            if (EditorGUI.EndChangeCheck() && m_AutoRebindProperty.boolValue)
            {
                m_NeedsRebind = true;
            }

            EditorGUI.BeginChangeCheck();
            EditorGUILayout.PropertyField(m_RootBoneProperty, Contents.rootBoneLabel);
            if (EditorGUI.EndChangeCheck())
            {
                m_NeedsRebind = true;
            }

            var hasSpriteChanged = HasAnySpriteChanged();
            if (m_ReorderableList == null || hasSpriteChanged)
            {
                UpdateSpriteCache();
                InitializeBoneTransformArray();
                SetupReorderableList();
            }

            DoBoneFoldout();

            EditorGUILayout.BeginHorizontal();
            GUILayout.FlexibleSpace();

            EditorGUI.BeginDisabledGroup(!EnableCreateBones());
            DoGenerateBonesButton();
            EditorGUI.EndDisabledGroup();

            EditorGUI.BeginDisabledGroup(!EnableSetBindPose());
            DoResetBindPoseButton();
            EditorGUI.EndDisabledGroup();

            GUILayout.FlexibleSpace();
            EditorGUILayout.EndHorizontal();

            serializedObject.ApplyModifiedProperties();

            if (m_NeedsRebind)
                Rebind();

            if (hasSpriteChanged && !m_SpriteSkins[0].ignoreNextSpriteChange)
            {
                ResetBounds(Undo.GetCurrentGroupName());
                m_SpriteSkins[0].ignoreNextSpriteChange = false;
            }

            DoValidationWarnings();
        }

        bool HasAnySpriteChanged()
        {
            for (var i = 0; i < m_SpriteSkins.Length; ++i)
            {
                var sprite = m_SpriteSkins[i].sprite;
                if (m_CurrentSprites[i] != sprite)
                    return true;
            }

            return false;
        }

        void DoBoneFoldout()
        {
            EditorGUILayout.Space();

            m_BoneFold = EditorGUILayout.Foldout(m_BoneFold, Contents.listHeaderLabel, true);
            if (m_BoneFold)
            {
                EditorGUI.BeginDisabledGroup(m_SpriteSkins[0].rootBone == null || m_BoneTransformsProperty.hasMultipleDifferentValues);
                m_ReorderableList.DoLayoutList();
                EditorGUI.EndDisabledGroup();
            }
        }

        void InitializeBoneTransformArray()
        {
            var hasSameNumberOfBones = true;
            var noOfBones = -1;
            for (var i = 0; i < m_SpriteSkins.Length; ++i)
            {
                if (m_SpriteSkins[i] == null || m_CurrentSprites[i] == null)
                    continue;
                if (i == 0)
                    noOfBones = m_CurrentSprites[i].GetBones().Length;
                else if (m_CurrentSprites[i].GetBones().Length != noOfBones)
                {
                    hasSameNumberOfBones = false;
                    break;
                }
            }

            if (hasSameNumberOfBones)
            {
                var elementCount = m_BoneTransformsProperty.arraySize;
                var bones = m_CurrentSprites[0] != null ? m_CurrentSprites[0].GetBones() : new SpriteBone[0];

                if (elementCount != bones.Length)
                {
                    m_BoneTransformsProperty.arraySize = bones.Length;

                    for (var i = elementCount; i < m_BoneTransformsProperty.arraySize; ++i)
                        m_BoneTransformsProperty.GetArrayElementAtIndex(i).objectReferenceValue = null;

                    m_NeedsRebind = true;
                }
            }
        }

        void SetupReorderableList()
        {
            m_ReorderableList = new ReorderableList(serializedObject, m_BoneTransformsProperty, false, true, false, false);
            m_ReorderableList.headerHeight = 1.0f;
            m_ReorderableList.elementHeightCallback = _ => EditorGUIUtility.singleLineHeight + 6;
            m_ReorderableList.drawElementCallback = (rect, index, _, _) =>
            {
                var content = GUIContent.none;

                if (m_CurrentSprites[0] != null)
                {
                    var bones = m_CurrentSprites[0].GetBones();
                    if (index < bones.Length)
                        content = new GUIContent(bones[index].name);
                }

                rect.y += 2f;
                rect.height = EditorGUIUtility.singleLineHeight;
                var element = m_BoneTransformsProperty.GetArrayElementAtIndex(index);

                EditorGUI.showMixedValue = m_BoneTransformsProperty.hasMultipleDifferentValues;
                EditorGUI.PropertyField(rect, element, content);
            };
        }


        void Rebind()
        {
            foreach (var skin in m_SpriteSkins)
            {
                if (skin.sprite == null || skin.rootBone == null)
                    continue;
                if (!SpriteSkinHelpers.GetSpriteBonesTransforms(skin, out var transforms))
                    Debug.LogWarning($"Rebind failed for {skin.name}. Could not find all bones required by the Sprite: {skin.sprite.name}.");
                skin.SetBoneTransforms(transforms);

                ResetBoundsIfNeeded(skin);
            }

            serializedObject.Update();
            m_NeedsRebind = false;
        }

        void ResetBounds(string undoName = "Reset Bounds")
        {
            foreach (var skin in m_SpriteSkins)
            {
                if (!skin.isValid)
                    continue;

                Undo.RegisterCompleteObjectUndo(skin, undoName);
                skin.CalculateBounds();

                EditorUtility.SetDirty(skin);
            }
        }

        static void ResetBoundsIfNeeded(SpriteSkin spriteSkin)
        {
            if (spriteSkin.isValid && spriteSkin.bounds == new Bounds())
                spriteSkin.CalculateBounds();
        }

        bool EnableCreateBones()
        {
            foreach (var skin in m_SpriteSkins)
            {
                var sprite = skin.sprite;
                if (sprite != null && skin.rootBone == null)
                    return true;
            }

            return false;
        }

        bool EnableSetBindPose()
        {
            return IsAnyTargetValid();
        }

        bool IsAnyTargetValid()
        {
            foreach (var skin in m_SpriteSkins)
            {
                if (skin.isValid)
                    return true;
            }

            return false;
        }

        void DoGenerateBonesButton()
        {
            if (GUILayout.Button("Create Bones", GUILayout.MaxWidth(125f)))
            {
                foreach (var skin in m_SpriteSkins)
                {
                    var sprite = skin.sprite;
                    if (sprite == null || skin.rootBone != null)
                        continue;

                    Undo.RegisterCompleteObjectUndo(skin, "Create Bones");

                    skin.CreateBoneHierarchy();

                    foreach (var transform in skin.boneTransforms)
                        Undo.RegisterCreatedObjectUndo(transform.gameObject, "Create Bones");

                    ResetBoundsIfNeeded(skin);

                    EditorUtility.SetDirty(skin);
                }

                serializedObject.SetIsDifferentCacheDirty();
                BoneGizmo.instance.boneGizmoController.OnSelectionChanged();
            }
        }

        void DoResetBindPoseButton()
        {
            if (GUILayout.Button("Reset Bind Pose", GUILayout.MaxWidth(125f)))
            {
                foreach (var skin in m_SpriteSkins)
                {
                    if (!skin.isValid)
                        continue;

                    Undo.RecordObjects(skin.boneTransforms, "Reset Bind Pose");
                    skin.ResetBindPose();
                }
            }
        }

        void DoValidationWarnings()
        {
            EditorGUILayout.Space();

            var preAppendObjectName = targets.Length > 1;

            foreach (var skin in m_SpriteSkins)
            {
                var validationResult = skin.Validate();
                if (validationResult == SpriteSkinState.Ready)
                    continue;

                var text = "";
                switch (validationResult)
                {
                    case SpriteSkinState.SpriteNotFound:
                        text = Contents.spriteNotFound;
                        break;
                    case SpriteSkinState.SpriteHasNoSkinningInformation:
                        text = Contents.spriteHasNoSkinningInformation;
                        break;
                    case SpriteSkinState.SpriteHasNoWeights:
                        text = Contents.spriteHasNoWeights;
                        break;
                    case SpriteSkinState.RootTransformNotFound:
                        text = Contents.rootTransformNotFound;
                        break;
                    case SpriteSkinState.InvalidTransformArray:
                        text = Contents.invalidTransformArray;
                        break;
                    case SpriteSkinState.InvalidTransformArrayLength:
                        text = Contents.invalidTransformArrayLength;
                        break;
                    case SpriteSkinState.TransformArrayContainsNull:
                        text = Contents.transformArrayContainsNull;
                        break;
                    case SpriteSkinState.InvalidBoneWeights:
                        text = Contents.invalidBoneWeights;
                        break;
                }

                if (preAppendObjectName)
                    text = $"{skin.name}:{text}";

                EditorGUILayout.HelpBox(text, MessageType.Warning);
            }
        }
    }
}
