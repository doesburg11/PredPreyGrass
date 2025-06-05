#pragma warning disable 0219

using System;
using System.Collections.Generic;
using UnityEditor.AssetImporters;
using UnityEditor.U2D.Aseprite.Common;
using UnityEditor.UIElements;
using UnityEngine;
using UnityEngine.UIElements;

namespace UnityEditor.U2D.Aseprite
{
    [CustomEditor(typeof(AsepriteImporter))]
    [CanEditMultipleObjects]
    internal class AsepriteImporterEditor : ScriptedImporterEditor, ITexturePlatformSettingsDataProvider
    {
        const string k_BaseFieldAlignedUssClass = "unity-base-field__aligned";
        const string k_HiddenElementUssClass = "HiddenElement";
        const string k_PaddingElementUssClass = "PaddingElement";
        const string k_SubElementUssClass = "SubElement";
        const string k_SubSubElementUssClass = "SubSubElement";

        // This number is in milliseconds. 
        const long k_PollForChangesInternal = 50;

        SerializedProperty m_GeneratePhysicsShape;
        SerializedProperty m_TextureType;
        SerializedProperty m_TextureShape;
        SerializedProperty m_SpriteMode;
        SerializedProperty m_SpritePixelsToUnits;
        SerializedProperty m_SpriteMeshType;
        SerializedProperty m_SpriteExtrude;
        SerializedProperty m_Alignment;
        SerializedProperty m_SpritePivot;
        SerializedProperty m_NPOTScale;
        SerializedProperty m_IsReadable;
        SerializedProperty m_sRGBTexture;
        SerializedProperty m_AlphaSource;
#if ENABLE_TEXTURE_STREAMING
        SerializedProperty m_StreamingMipmaps;
        SerializedProperty m_StreamingMipmapsPriority;
#endif
        SerializedProperty m_MipMapMode;
        SerializedProperty m_EnableMipMap;
        SerializedProperty m_FadeOut;
        SerializedProperty m_BorderMipMap;
        SerializedProperty m_MipMapsPreserveCoverage;
        SerializedProperty m_AlphaTestReferenceValue;
        SerializedProperty m_MipMapFadeDistanceStart;
        SerializedProperty m_MipMapFadeDistanceEnd;
        SerializedProperty m_AlphaIsTransparency;
        SerializedProperty m_FilterMode;
        SerializedProperty m_Aniso;

        SerializedProperty m_WrapU;
        SerializedProperty m_WrapV;
        SerializedProperty m_WrapW;
        SerializedProperty m_ConvertToNormalMap;
        SerializedProperty m_PlatformSettingsArrProp;

        SerializedProperty m_FileImportMode;
        SerializedProperty m_ImportHiddenLayers;
        SerializedProperty m_LayerImportMode;
        SerializedProperty m_DefaultPivotSpace;
        SerializedProperty m_DefaultPivotAlignment;
        SerializedProperty m_CustomPivotPosition;
        SerializedProperty m_MosaicPadding;
        SerializedProperty m_SpritePadding;

        SerializedProperty m_GenerateModelPrefab;
        SerializedProperty m_AddSortingGroup;
        SerializedProperty m_AddShadowCasters;
        SerializedProperty m_GenerateAnimationClips;
        SerializedProperty m_PrevGenerateAnimationClips;
        SerializedProperty m_GenerateIndividualEvents;
        SerializedProperty m_GenerateSpriteAtlas;

        VisualElement m_RootVisualElement;
        VisualElement m_InspectorSettingsView;
        
        readonly AsepriteImporterEditorFoldOutState m_EditorFoldOutState = new();
        bool m_ShowPerAxisWrapModes = false;
        readonly int[] m_FilterModeOptions = (int[])(Enum.GetValues(typeof(FilterMode)));
        TexturePlatformSettingsHelper m_TexturePlatformSettingsHelper;
        VisualElement[] m_InspectorUI;
        int m_ActiveEditorIndex = 0;

        bool ITexturePlatformSettingsDataProvider.textureTypeHasMultipleDifferentValues => m_TextureType.hasMultipleDifferentValues;
        TextureImporterType ITexturePlatformSettingsDataProvider.textureType => (TextureImporterType)m_TextureType.intValue;
        SpriteImportMode ITexturePlatformSettingsDataProvider.spriteImportMode => spriteImportMode;

        FileImportModes fileImportMode => (FileImportModes)m_FileImportMode.intValue;
        SpriteImportMode spriteImportMode => (SpriteImportMode)m_SpriteMode.intValue;

        AnimationClip m_DefaultClip;
        ModelPreviewer m_ModelPreviewer;
        AsepriteImporter[] m_ImporterTargets;
        string[] m_AssetPaths;

        /// <summary>
        /// The SerializedProperty of an array of TextureImporterPlatformSettings.
        /// </summary>
        public SerializedProperty platformSettingsArray => m_PlatformSettingsArrProp;

        public override bool showImportedObject => false;
        public override bool UseDefaultMargins() => false;

        /// <summary>
        /// Implementation of AssetImporterEditor.OnEnable
        /// </summary>
        public override void OnEnable()
        {
            base.OnEnable();

            CacheImporterData();
            CacheSerializedProperties();
            SetupInspectorUI();
            InitPreview();
        }

        void CacheImporterData()
        {
            m_ImporterTargets = new AsepriteImporter[targets.Length];
            m_AssetPaths = new string[targets.Length];
            for (var i = 0; i < targets.Length; ++i)
            {
                var importer = (AsepriteImporter)targets[i];
                m_ImporterTargets[i] = importer;
                m_AssetPaths[i] = importer.assetPath;
            }
        }

        void CacheSerializedProperties()
        {
            m_GeneratePhysicsShape = serializedObject.FindProperty("m_GeneratePhysicsShape");

            var textureImporterSettingsSP = serializedObject.FindProperty("m_TextureImporterSettings");
            m_TextureType = textureImporterSettingsSP.FindPropertyRelative("m_TextureType");
            m_TextureShape = textureImporterSettingsSP.FindPropertyRelative("m_TextureShape");
            m_ConvertToNormalMap = textureImporterSettingsSP.FindPropertyRelative("m_ConvertToNormalMap");
            m_SpriteMode = textureImporterSettingsSP.FindPropertyRelative("m_SpriteMode");
            m_SpritePixelsToUnits = textureImporterSettingsSP.FindPropertyRelative("m_SpritePixelsToUnits");
            m_SpriteMeshType = textureImporterSettingsSP.FindPropertyRelative("m_SpriteMeshType");
            m_SpriteExtrude = textureImporterSettingsSP.FindPropertyRelative("m_SpriteExtrude");
            m_Alignment = textureImporterSettingsSP.FindPropertyRelative("m_Alignment");
            m_SpritePivot = textureImporterSettingsSP.FindPropertyRelative("m_SpritePivot");
            m_NPOTScale = textureImporterSettingsSP.FindPropertyRelative("m_NPOTScale");
            m_IsReadable = textureImporterSettingsSP.FindPropertyRelative("m_IsReadable");
            m_sRGBTexture = textureImporterSettingsSP.FindPropertyRelative("m_sRGBTexture");
            m_AlphaSource = textureImporterSettingsSP.FindPropertyRelative("m_AlphaSource");
#if ENABLE_TEXTURE_STREAMING
            m_StreamingMipmaps = textureImporterSettingsSP.FindPropertyRelative("m_StreamingMipmaps");
            m_StreamingMipmapsPriority = textureImporterSettingsSP.FindPropertyRelative("m_StreamingMipmapsPriority");
#endif
            m_MipMapMode = textureImporterSettingsSP.FindPropertyRelative("m_MipMapMode");
            m_EnableMipMap = textureImporterSettingsSP.FindPropertyRelative("m_EnableMipMap");
            m_FadeOut = textureImporterSettingsSP.FindPropertyRelative("m_FadeOut");
            m_BorderMipMap = textureImporterSettingsSP.FindPropertyRelative("m_BorderMipMap");
            m_MipMapsPreserveCoverage = textureImporterSettingsSP.FindPropertyRelative("m_MipMapsPreserveCoverage");
            m_AlphaTestReferenceValue = textureImporterSettingsSP.FindPropertyRelative("m_AlphaTestReferenceValue");
            m_MipMapFadeDistanceStart = textureImporterSettingsSP.FindPropertyRelative("m_MipMapFadeDistanceStart");
            m_MipMapFadeDistanceEnd = textureImporterSettingsSP.FindPropertyRelative("m_MipMapFadeDistanceEnd");
            m_AlphaIsTransparency = textureImporterSettingsSP.FindPropertyRelative("m_AlphaIsTransparency");
            m_FilterMode = textureImporterSettingsSP.FindPropertyRelative("m_FilterMode");
            m_Aniso = textureImporterSettingsSP.FindPropertyRelative("m_Aniso");
            m_WrapU = textureImporterSettingsSP.FindPropertyRelative("m_WrapU");
            m_WrapV = textureImporterSettingsSP.FindPropertyRelative("m_WrapV");
            m_WrapW = textureImporterSettingsSP.FindPropertyRelative("m_WrapW");
            m_PlatformSettingsArrProp = extraDataSerializedObject.FindProperty("platformSettings");

            var asepriteImporterSettings = serializedObject.FindProperty("m_AsepriteImporterSettings");
            m_FileImportMode = asepriteImporterSettings.FindPropertyRelative("m_FileImportMode");
            m_ImportHiddenLayers = asepriteImporterSettings.FindPropertyRelative("m_ImportHiddenLayers");
            m_LayerImportMode = asepriteImporterSettings.FindPropertyRelative("m_LayerImportMode");
            m_DefaultPivotSpace = asepriteImporterSettings.FindPropertyRelative("m_DefaultPivotSpace");
            m_DefaultPivotAlignment = asepriteImporterSettings.FindPropertyRelative("m_DefaultPivotAlignment");
            m_CustomPivotPosition = asepriteImporterSettings.FindPropertyRelative("m_CustomPivotPosition");
            m_MosaicPadding = asepriteImporterSettings.FindPropertyRelative("m_MosaicPadding");
            m_SpritePadding = asepriteImporterSettings.FindPropertyRelative("m_SpritePadding");

            m_GenerateModelPrefab = asepriteImporterSettings.FindPropertyRelative("m_GenerateModelPrefab");
            m_AddSortingGroup = asepriteImporterSettings.FindPropertyRelative("m_AddSortingGroup");
            m_AddShadowCasters = asepriteImporterSettings.FindPropertyRelative("m_AddShadowCasters");
            m_GenerateAnimationClips = asepriteImporterSettings.FindPropertyRelative("m_GenerateAnimationClips");
            m_GenerateIndividualEvents = asepriteImporterSettings.FindPropertyRelative("m_GenerateIndividualEvents");
            m_GenerateSpriteAtlas = asepriteImporterSettings.FindPropertyRelative("m_GenerateSpriteAtlas");

            var prevAsepriteImporterSettings = serializedObject.FindProperty("m_PreviousAsepriteImporterSettings");
            m_PrevGenerateAnimationClips = prevAsepriteImporterSettings.FindPropertyRelative("m_GenerateAnimationClips");

            m_TexturePlatformSettingsHelper = new TexturePlatformSettingsHelper(this);
        }

        void SetupInspectorUI()
        {
            m_InspectorUI = new[]
            {
                SetupSpriteActorContainer()
            };
            m_ActiveEditorIndex = Mathf.Max(EditorPrefs.GetInt(this.GetType().Name + "ActiveEditorIndex", 0), 0);
            m_ActiveEditorIndex %= m_InspectorUI.Length;
        }

        VisualElement SetupSpriteActorContainer()
        {
            var spriteActorContainer = new VisualElement()
            {
                name = "SpriteActorContainer"
            };

            SetupGeneralContainer(spriteActorContainer);
            SetupLayerImportContainer(spriteActorContainer);
            SetupGenerateAssetContainer(spriteActorContainer);
            SetupCommonTextureSettingsContainer(spriteActorContainer);
            SetupPlatformSettingsContainer(spriteActorContainer);
            SetupAdvancedContainer(spriteActorContainer);

            var applyRevertContainer = new IMGUIContainer(ApplyRevertGUI);
            spriteActorContainer.Add(applyRevertContainer);

            return spriteActorContainer;
        }

        void SetupGeneralContainer(VisualElement root)
        {
            var foldOut = new Foldout()
            {
                text = styles.generalHeaderText.text,
                tooltip = styles.generalHeaderText.tooltip,
                value = m_EditorFoldOutState.generalFoldout
            };
            ImporterEditorUtils.AddSkinUssClass(foldOut.Q<Toggle>());
            foldOut.RegisterValueChangedCallback(_ => { m_EditorFoldOutState.generalFoldout = foldOut.value; });
            root.Add(foldOut);

            var importModeField = new PropertyField(m_FileImportMode, styles.fileImportMode.text)
            {
                tooltip = styles.fileImportMode.tooltip
            };
            importModeField.Bind(serializedObject);
            foldOut.Add(importModeField);

            var ppuField = new PropertyField(m_SpritePixelsToUnits, styles.spritePixelsPerUnit.text)
            {
                tooltip = styles.spritePixelsPerUnit.tooltip
            };
            ppuField.Bind(serializedObject);
            ppuField.RegisterValueChangeCallback(x =>
            {
                m_SpritePixelsToUnits.floatValue = Mathf.Max(m_SpritePixelsToUnits.floatValue, 1f);
                serializedObject.ApplyModifiedProperties();
            });
            foldOut.Add(ppuField);

            var meshField = new PopupField<string>(styles.spriteMeshTypeOptions, m_SpriteMeshType.intValue)
            {
                label = styles.spriteMeshType.text,
                tooltip = styles.spriteMeshType.tooltip
            };
            meshField.AddToClassList(k_BaseFieldAlignedUssClass);
            meshField.RegisterValueChangedCallback(x =>
            {
                m_SpriteMeshType.intValue = meshField.index;
                serializedObject.ApplyModifiedProperties();
            });
            foldOut.Add(meshField);

            var physicsShapeField = new PropertyField(m_GeneratePhysicsShape, styles.generatePhysicsShape.text)
            {
                tooltip = styles.generatePhysicsShape.tooltip
            };
            physicsShapeField.Bind(serializedObject);
            foldOut.Add(physicsShapeField);

            SetupSpriteEditorButton(foldOut);

            var paddingElement = new VisualElement()
            {
                name = "PaddingElement"
            };
            paddingElement.AddToClassList(k_PaddingElementUssClass);
            foldOut.Add(paddingElement);
        }

        void SetupSpriteEditorButton(VisualElement root)
        {
            var spriteEditorBtn = new Button()
            {
                text = styles.spriteEditorButtonLabel.text
            };
            root.Add(spriteEditorBtn);

            spriteEditorBtn.SetEnabled(CanOpenSpriteEditor());
            spriteEditorBtn.clicked += () =>
            {
                if (HasModified())
                {
                    // To ensure Sprite Editor Window to have the latest texture import setting,
                    // We must applied those modified values first.
                    var dialogText = string.Format(s_Styles.unappliedSettingsDialogContent.text, ((AssetImporter)target).assetPath);
                    if (EditorUtility.DisplayDialog(s_Styles.unappliedSettingsDialogTitle.text,
                            dialogText, s_Styles.applyButtonLabel.text, s_Styles.cancelButtonLabel.text))
                    {
                        SaveChanges();
                        InternalEditorBridge.ShowSpriteEditorWindow(this.assetTarget);

                        // We re-imported the asset which destroyed the editor, so we can't keep running the UI here.
                        GUIUtility.ExitGUI();
                    }
                }
                else
                {
                    InternalEditorBridge.ShowSpriteEditorWindow(this.assetTarget);
                }
            };
        }

        bool CanOpenSpriteEditor()
        {
            return m_ImporterTargets.Length == 1 && m_ImporterTargets[0].textureActualWidth > 0;
        }

        void SetupLayerImportContainer(VisualElement root)
        {
            var foldOut = new Foldout()
            {
                text = styles.layerImportHeaderText.text,
                tooltip = styles.layerImportHeaderText.tooltip,
                value = m_EditorFoldOutState.layerImportFoldout
            };
            ImporterEditorUtils.AddSkinUssClass(foldOut.Q<Toggle>());
            foldOut.RegisterValueChangedCallback(_ => { m_EditorFoldOutState.layerImportFoldout = foldOut.value; });
            root.Add(foldOut);

            var hiddenLayersField = new PropertyField(m_ImportHiddenLayers, styles.importHiddenLayer.text)
            {
                tooltip = styles.importHiddenLayer.tooltip
            };
            hiddenLayersField.schedule.Execute(() =>
            {
                var shouldShow = fileImportMode is FileImportModes.AnimatedSprite or FileImportModes.SpriteSheet;
                if (hiddenLayersField.visible != shouldShow)
                {
                    hiddenLayersField.visible = shouldShow;
                    hiddenLayersField.EnableInClassList(k_HiddenElementUssClass, !shouldShow);
                }
            }).Every(k_PollForChangesInternal);             
            hiddenLayersField.Bind(serializedObject);
            foldOut.Add(hiddenLayersField);

            var layerModePopup = new PopupField<string>(s_Styles.layerImportOptions, m_LayerImportMode.intValue)
            {
                label = s_Styles.layerImportMode.text,
                tooltip = s_Styles.layerImportMode.tooltip
            };
            layerModePopup.RegisterValueChangedCallback(_ =>
            {
                m_LayerImportMode.intValue = layerModePopup.index;
                serializedObject.ApplyModifiedProperties();
            });
            layerModePopup.schedule.Execute(() =>
            {
                var shouldShow = fileImportMode is FileImportModes.AnimatedSprite;
                if (layerModePopup.visible != shouldShow)
                {
                    layerModePopup.visible = shouldShow;
                    layerModePopup.EnableInClassList(k_HiddenElementUssClass, !shouldShow);
                }
            }).Every(k_PollForChangesInternal);            
            layerModePopup.AddToClassList(k_BaseFieldAlignedUssClass);
            foldOut.Add(layerModePopup);

            var pivotSpaceField = new PropertyField(m_DefaultPivotSpace, styles.defaultPivotSpace.text)
            {
                tooltip = styles.defaultPivotSpace.tooltip
            };
            pivotSpaceField.Bind(serializedObject);
            pivotSpaceField.schedule.Execute(() =>
            {
                var shouldShow = fileImportMode == FileImportModes.AnimatedSprite;
                if (pivotSpaceField.visible != shouldShow)
                {
                    pivotSpaceField.visible = shouldShow;
                    pivotSpaceField.EnableInClassList(k_HiddenElementUssClass, !shouldShow);
                }
            }).Every(k_PollForChangesInternal);
            foldOut.Add(pivotSpaceField);

            var pivotAlignmentPopup = new PopupField<string>(s_Styles.spriteAlignmentOptions, m_DefaultPivotAlignment.intValue)
            {
                label = s_Styles.defaultPivotAlignment.text,
                tooltip = s_Styles.defaultPivotAlignment.tooltip
            };
            pivotAlignmentPopup.RegisterValueChangedCallback(x =>
            {
                m_DefaultPivotAlignment.intValue = pivotAlignmentPopup.index;
                serializedObject.ApplyModifiedProperties();
            });
            pivotAlignmentPopup.AddToClassList(k_BaseFieldAlignedUssClass);
            pivotAlignmentPopup.schedule.Execute(() =>
            {
                var shouldShow = fileImportMode == FileImportModes.AnimatedSprite;
                if (pivotAlignmentPopup.visible != shouldShow)
                {
                    pivotAlignmentPopup.visible = shouldShow;
                    pivotAlignmentPopup.EnableInClassList(k_HiddenElementUssClass, !shouldShow);
                }
            }).Every(k_PollForChangesInternal);
            foldOut.Add(pivotAlignmentPopup);

            var shouldShow = (SpriteAlignment)pivotAlignmentPopup.index == SpriteAlignment.Custom;
            var customPivotField = new PropertyField(m_CustomPivotPosition, styles.customPivotPosition.text)
            {
                tooltip = styles.customPivotPosition.tooltip,
                visible = shouldShow
            };
            customPivotField.Bind(serializedObject);
            customPivotField.EnableInClassList(k_HiddenElementUssClass, !shouldShow);
            customPivotField.schedule.Execute(x =>
            {
                var isShowing = (SpriteAlignment)pivotAlignmentPopup.index == SpriteAlignment.Custom;
                if (customPivotField.visible != isShowing)
                {
                    customPivotField.visible = isShowing;
                    customPivotField.EnableInClassList(k_HiddenElementUssClass, !isShowing);
                }
            }).Every(k_PollForChangesInternal);
            foldOut.Add(customPivotField);

            var mosaicPaddingField = new PropertyField(m_MosaicPadding, styles.mosaicPadding.text)
            {
                tooltip = styles.mosaicPadding.tooltip
            };
            mosaicPaddingField.Bind(serializedObject);
            foldOut.Add(mosaicPaddingField);

            var spritePaddingField = new PropertyField(m_SpritePadding, styles.spritePadding.text)
            {
                tooltip = styles.spritePadding.tooltip
            };
            spritePaddingField.Bind(serializedObject);
            spritePaddingField.schedule.Execute(() =>
            {
                var isShowing = fileImportMode == FileImportModes.AnimatedSprite;
                if (spritePaddingField.visible != isShowing)
                {
                    spritePaddingField.visible = isShowing;
                    spritePaddingField.EnableInClassList(k_HiddenElementUssClass, !isShowing);
                }
            }).Every(k_PollForChangesInternal);
            foldOut.Add(spritePaddingField);

            var paddingElement = new VisualElement()
            {
                name = "PaddingElement"
            };
            paddingElement.AddToClassList(k_PaddingElementUssClass);
            foldOut.Add(paddingElement);
        }

        void SetupGenerateAssetContainer(VisualElement root)
        {
            // Generate Assets foldout
            var foldOut = new Foldout()
            {
                text = styles.generateAssetsHeaderText.text,
                tooltip = styles.generateAssetsHeaderText.tooltip,
                value = m_EditorFoldOutState.generateAssetFoldout
            };
            ImporterEditorUtils.AddSkinUssClass(foldOut.Q<Toggle>());
            foldOut.RegisterValueChangedCallback(_ => { m_EditorFoldOutState.generateAssetFoldout = foldOut.value; });
            foldOut.schedule.Execute(() =>
            {
                var shouldShow = fileImportMode is FileImportModes.AnimatedSprite; //or FileImportModes.TileSet;
                if (foldOut.visible != shouldShow)
                {
                    foldOut.visible = shouldShow;
                    foldOut.EnableInClassList(k_HiddenElementUssClass, !shouldShow);
                }
            }).Every(k_PollForChangesInternal);
            root.Add(foldOut);

            var animatedSpritesParent = new VisualElement();
            animatedSpritesParent.schedule.Execute(() =>
            {
                var shouldShow = fileImportMode is FileImportModes.AnimatedSprite;
                if (animatedSpritesParent.visible != shouldShow)
                {
                    animatedSpritesParent.visible = shouldShow;
                    animatedSpritesParent.EnableInClassList(k_HiddenElementUssClass, !shouldShow);
                }
            }).Every(k_PollForChangesInternal);
            foldOut.Add(animatedSpritesParent);
            SetupGenerateAssetsForAnimatedSprites(animatedSpritesParent);

            /*
            var tileSetParent = new VisualElement();
            tileSetParent.schedule.Execute(() =>
            {
                var shouldShow = fileImportMode is FileImportModes.TileSet;
                if (tileSetParent.visible != shouldShow)
                {
                    tileSetParent.visible = shouldShow;
                    tileSetParent.EnableInClassList(k_HiddenElementUssClass, !shouldShow);
                }
            }).Every(k_PollForChangesInternal);            
            foldOut.Add(tileSetParent);
            SetupGeneratedAssetsForTileSet(tileSetParent);
            */

            // Footer element
            var paddingElement = new VisualElement()
            {
                name = "PaddingElement"
            };
            paddingElement.AddToClassList(k_PaddingElementUssClass);
            foldOut.Add(paddingElement);
        }

        void SetupGenerateAssetsForAnimatedSprites(VisualElement parent)
        {
#if ENABLE_URP
            const bool isUrpEnabled = true;
#else
            const bool isUrpEnabled = false;
#endif
            
            // Generate model prefab toggle
            var generateModelField = new PropertyField(m_GenerateModelPrefab, styles.generateModelPrefab.text)
            {
                tooltip = styles.generateModelPrefab.tooltip
            };
            generateModelField.Bind(serializedObject);
            parent.Add(generateModelField);

            // Add "sorting group"-component
            var isSortingEnabled = m_GenerateModelPrefab.boolValue;
            var sortingGroupField = new PropertyField(m_AddSortingGroup, styles.addSortingGroup.text)
            {
                tooltip = styles.addSortingGroup.tooltip
            };
            sortingGroupField.Bind(serializedObject);
            sortingGroupField.AddToClassList(k_SubElementUssClass);
            sortingGroupField.SetEnabled(isSortingEnabled);
            sortingGroupField.schedule.Execute(() =>
            {
                isSortingEnabled = m_GenerateModelPrefab.boolValue;
                if (sortingGroupField.enabledSelf != isSortingEnabled)
                    sortingGroupField.SetEnabled(isSortingEnabled);
            }).Every(k_PollForChangesInternal);
            parent.Add(sortingGroupField);
            
            // Add "shadow caster"-component
            var areShadowsEnabled = isUrpEnabled && m_GenerateModelPrefab.boolValue;
            var shadowCasterField = new PropertyField(m_AddShadowCasters, styles.addShadowCasters.text)
            {
                tooltip = styles.addShadowCasters.tooltip
            };
            shadowCasterField.Bind(serializedObject);
            shadowCasterField.AddToClassList(k_SubElementUssClass);
            shadowCasterField.SetEnabled(areShadowsEnabled);
            shadowCasterField.schedule.Execute(() =>
            {
                areShadowsEnabled = isUrpEnabled && m_GenerateModelPrefab.boolValue;
                if (shadowCasterField.enabledSelf != areShadowsEnabled)
                    shadowCasterField.SetEnabled(areShadowsEnabled);
            }).Every(k_PollForChangesInternal);
            parent.Add(shadowCasterField);

            // Generate animation clips toggle
            var generateClipsField = new PropertyField(m_GenerateAnimationClips, styles.generateAnimationClips.text)
            {
                tooltip = styles.generateAnimationClips.tooltip
            };
            generateClipsField.Bind(serializedObject);
            parent.Add(generateClipsField);
            
            // Generate individual animation events toggle
            var isIndividualEventsEnabled = m_GenerateAnimationClips.boolValue;
            var generateIndividualEventsField = new PropertyField(m_GenerateIndividualEvents, styles.generateIndividualEvents.text)
            {
                tooltip = styles.generateIndividualEvents.tooltip
            };
            generateIndividualEventsField.Bind(serializedObject);
            generateIndividualEventsField.AddToClassList(k_SubElementUssClass);
            generateIndividualEventsField.SetEnabled(isIndividualEventsEnabled);
            generateIndividualEventsField.schedule.Execute(() =>
            {
                isIndividualEventsEnabled = m_GenerateAnimationClips.boolValue;
                if (generateIndividualEventsField.enabledSelf != isIndividualEventsEnabled)
                    generateIndividualEventsField.SetEnabled(isIndividualEventsEnabled);
            }).Every(k_PollForChangesInternal);
            parent.Add(generateIndividualEventsField);            

            SetupAnimationAssetsButton(parent);
        }

        void SetupGeneratedAssetsForTileSet(VisualElement parent)
        {
            var generateSpriteAtlasField = new PropertyField(m_GenerateSpriteAtlas, styles.generateSpriteAtlas.text)
            {
                tooltip = styles.generateSpriteAtlas.tooltip
            };
            generateSpriteAtlasField.Bind(serializedObject);
            parent.Add(generateSpriteAtlasField);           
        }

        void SetupAnimationAssetsButton(VisualElement root)
        {
            var isEnabled = m_ImporterTargets.Length == 1 &&
                            m_DefaultClip != null &&
                            m_GenerateAnimationClips.boolValue;

            var assetsBtn = new Button()
            {
                text = styles.exportAnimationAssetsText.text
            };
            assetsBtn.SetEnabled(isEnabled);
            assetsBtn.schedule.Execute(() =>
            {
                isEnabled = m_ImporterTargets.Length == 1 &&
                            m_DefaultClip != null &&
                            m_GenerateAnimationClips.boolValue;
                if (assetsBtn.enabledSelf != isEnabled)
                    assetsBtn.SetEnabled(isEnabled);
            }).Every(k_PollForChangesInternal);
            assetsBtn.clicked += () =>
            {
                var window = EditorWindow.GetWindow<ExportAssetsPopup>();
                window.ShowExportPopup(this, m_ImporterTargets);
            };
            root.Add(assetsBtn);

            var helpBox = new HelpBox(styles.exportAnimationInfoText.text, HelpBoxMessageType.Info);
            helpBox.visible = m_GenerateAnimationClips.boolValue && !m_PrevGenerateAnimationClips.boolValue;
            helpBox.EnableInClassList(k_HiddenElementUssClass, !helpBox.visible);
            helpBox.schedule.Execute(() =>
            {
                // If the Generate Animation Clips checkbox has been checked, but not yet applied.
                var isVisible = m_GenerateAnimationClips.boolValue && !m_PrevGenerateAnimationClips.boolValue;
                if (helpBox.visible != isVisible)
                {
                    helpBox.visible = isVisible;
                    helpBox.EnableInClassList(k_HiddenElementUssClass, !isVisible);
                }
            }).Every(k_PollForChangesInternal);
            root.Add(helpBox);
        }

        void SetupCommonTextureSettingsContainer(VisualElement root)
        {
            var foldOut = new Foldout()
            {
                text = styles.textureHeaderText.text,
                tooltip = styles.textureHeaderText.tooltip,
                value = m_EditorFoldOutState.textureFoldout
            };
            ImporterEditorUtils.AddSkinUssClass(foldOut.Q<Toggle>());
            foldOut.RegisterValueChangedCallback(_ => { m_EditorFoldOutState.textureFoldout = foldOut.value; });
            root.Add(foldOut);

            var imguiContainer = new IMGUIContainer(() =>
            {
                serializedObject.Update();
                extraDataSerializedObject.Update();

                EditorGUI.BeginChangeCheck();

                // Wrap mode
                var isVolume = false;
                WrapModePopup(m_WrapU, m_WrapV, m_WrapW, isVolume, ref m_ShowPerAxisWrapModes);

                // Display warning about repeat wrap mode on restricted npot emulation
                if (m_NPOTScale.intValue == (int)TextureImporterNPOTScale.None &&
                    (m_WrapU.intValue == (int)TextureWrapMode.Repeat || m_WrapV.intValue == (int)TextureWrapMode.Repeat) &&
                    !InternalEditorBridge.DoesHardwareSupportsFullNPOT())
                {
                    var displayWarning = false;
                    foreach (var importer in m_ImporterTargets)
                    {
                        var w = importer.textureActualWidth;
                        var h = importer.textureActualHeight;
                        if (!Mathf.IsPowerOfTwo(w) || !Mathf.IsPowerOfTwo(h))
                        {
                            displayWarning = true;
                            break;
                        }
                    }

                    if (displayWarning)
                    {
                        EditorGUILayout.HelpBox(styles.warpNotSupportWarning.text, MessageType.Warning, true);
                    }
                }

                // Filter mode
                EditorGUI.showMixedValue = m_FilterMode.hasMultipleDifferentValues;
                var filter = (FilterMode)m_FilterMode.intValue;
                if ((int)filter == -1)
                {
                    if (m_FadeOut.intValue > 0 || m_ConvertToNormalMap.intValue > 0)
                        filter = FilterMode.Trilinear;
                    else
                        filter = FilterMode.Bilinear;
                }
                filter = (FilterMode)EditorGUILayout.IntPopup(styles.filterMode, (int)filter, styles.filterModeOptions, m_FilterModeOptions);
                EditorGUI.showMixedValue = false;
                if (EditorGUI.EndChangeCheck())
                {
                    m_FilterMode.intValue = (int) filter;
                    m_FilterMode.serializedObject.ApplyModifiedProperties();
                }

                // Aniso
                var showAniso = (FilterMode)m_FilterMode.intValue != FilterMode.Point
                    && m_EnableMipMap.intValue > 0
                    && (TextureImporterShape)m_TextureShape.intValue != TextureImporterShape.TextureCube;
                using (new EditorGUI.DisabledScope(!showAniso))
                {
                    EditorGUI.BeginChangeCheck();
                    EditorGUI.showMixedValue = m_Aniso.hasMultipleDifferentValues;
                    int aniso = m_Aniso.intValue;
                    if (aniso == -1)
                        aniso = 1;
                    aniso = EditorGUILayout.IntSlider(styles.anisoLevelLabel, aniso, 0, 16);
                    EditorGUI.showMixedValue = false;
                    if (EditorGUI.EndChangeCheck())
                    {
                        m_Aniso.intValue = aniso;
                        m_Aniso.serializedObject.ApplyModifiedProperties();
                    }

                    if (aniso > 1)
                    {
                        if (QualitySettings.anisotropicFiltering == AnisotropicFiltering.Disable)
                            EditorGUILayout.HelpBox(styles.anisotropicDisableInfo.text, MessageType.Info);
                        else if (QualitySettings.anisotropicFiltering == AnisotropicFiltering.ForceEnable)
                            EditorGUILayout.HelpBox(styles.anisotropicForceEnableInfo.text, MessageType.Info);
                    }
                }
                GUILayout.Space(5);

                serializedObject.ApplyModifiedProperties();
                extraDataSerializedObject.ApplyModifiedProperties();
            });
            foldOut.Add(imguiContainer);
        }

        void SetupPlatformSettingsContainer(VisualElement root)
        {
            var foldOut = new Foldout()
            {
                text = styles.platformSettingsHeaderText.text,
                tooltip = styles.platformSettingsHeaderText.tooltip,
                value = m_EditorFoldOutState.platformSettingsFoldout
            };
            ImporterEditorUtils.AddSkinUssClass(foldOut.Q<Toggle>());
            foldOut.RegisterValueChangedCallback(_ => { m_EditorFoldOutState.platformSettingsFoldout = foldOut.value; });
            root.Add(foldOut);

            var imguiContainer = new IMGUIContainer(() =>
            {
                serializedObject.Update();
                extraDataSerializedObject.Update();

                GUILayout.Space(5);
                m_TexturePlatformSettingsHelper.ShowPlatformSpecificSettings();
                GUILayout.Space(5);

                serializedObject.ApplyModifiedProperties();
                extraDataSerializedObject.ApplyModifiedProperties();
            });
            foldOut.Add(imguiContainer);
        }

        void SetupAdvancedContainer(VisualElement root)
        {
            if (m_TextureType.hasMultipleDifferentValues)
                return;

            var foldOut = new Foldout()
            {
                text = styles.advancedHeaderText.text,
                tooltip = styles.advancedHeaderText.tooltip,
                value = m_EditorFoldOutState.advancedFoldout
            };
            ImporterEditorUtils.AddSkinUssClass(foldOut.Q<Toggle>());
            foldOut.RegisterValueChangedCallback(_ => { m_EditorFoldOutState.advancedFoldout = foldOut.value; });
            root.Add(foldOut);
            
            // sRGB Toggle
            var srgbToggle = new Toggle(styles.sRGBTexture.text)
            {
                value = m_sRGBTexture.intValue > 0,
                tooltip = styles.sRGBTexture.tooltip,
            };
            srgbToggle.AddToClassList(k_BaseFieldAlignedUssClass);
            srgbToggle.RegisterValueChangedCallback(_ =>
            {
                m_sRGBTexture.intValue = srgbToggle.value ? 1 : 0;
                m_sRGBTexture.serializedObject.ApplyModifiedProperties();
            });
            foldOut.Add(srgbToggle);

            // AlphaSource Enum
            var alphaSourceEnum = new EnumField(styles.alphaSource.text, TextureImporterAlphaSource.FromInput)
            {
                value = (TextureImporterAlphaSource)m_AlphaSource.intValue,
                tooltip = styles.alphaSource.tooltip
            };
            alphaSourceEnum.AddToClassList(k_BaseFieldAlignedUssClass);
            alphaSourceEnum.RegisterValueChangedCallback(_ =>
            {
                m_AlphaSource.intValue = (int)(TextureImporterAlphaSource)alphaSourceEnum.value;
                m_AlphaSource.serializedObject.ApplyModifiedProperties();
            });
            foldOut.Add(alphaSourceEnum);       
            
            // AlphaIsTransparency Toggle
            var alphaIsTransparencyToggle = new Toggle(styles.alphaIsTransparency.text)
            {
                value = m_AlphaIsTransparency.intValue > 0,
                tooltip = styles.alphaIsTransparency.tooltip,
            };
            alphaIsTransparencyToggle.AddToClassList(k_BaseFieldAlignedUssClass);
            alphaIsTransparencyToggle.RegisterValueChangedCallback(_ =>
            {
                m_AlphaIsTransparency.intValue = alphaIsTransparencyToggle.value ? 1 : 0;
                m_AlphaIsTransparency.serializedObject.ApplyModifiedProperties();
            });
            var showAlphaIsTransparency = (TextureImporterAlphaSource)m_AlphaSource.intValue != TextureImporterAlphaSource.None;
            alphaIsTransparencyToggle.SetEnabled(showAlphaIsTransparency);
            alphaIsTransparencyToggle.schedule.Execute(() =>
            {
                showAlphaIsTransparency = (TextureImporterAlphaSource)m_AlphaSource.intValue != TextureImporterAlphaSource.None;
                if (alphaIsTransparencyToggle.enabledSelf != showAlphaIsTransparency)
                    alphaIsTransparencyToggle.SetEnabled(showAlphaIsTransparency);
            }).Every(k_PollForChangesInternal);            
            foldOut.Add(alphaIsTransparencyToggle);            
            
            // Read/Write Enabled Toggle
            var readWriteToggle = new Toggle(styles.readWrite.text)
            {
                value = m_IsReadable.intValue > 0,
                tooltip = styles.readWrite.tooltip,
            };
            readWriteToggle.AddToClassList(k_BaseFieldAlignedUssClass);
            readWriteToggle.RegisterValueChangedCallback(_ =>
            {
                m_IsReadable.intValue = readWriteToggle.value ? 1 : 0;
                m_IsReadable.serializedObject.ApplyModifiedProperties();
            });
            foldOut.Add(readWriteToggle);
            
            // MipMap Toggle
            var mipmapToggle = new Toggle(styles.generateMipMaps.text)
            {
                value = m_EnableMipMap.intValue > 0,
                tooltip = styles.generateMipMaps.tooltip,
            };
            mipmapToggle.AddToClassList(k_BaseFieldAlignedUssClass);
            mipmapToggle.RegisterValueChangedCallback(_ =>
            {
                m_EnableMipMap.intValue = mipmapToggle.value ? 1 : 0;
                m_EnableMipMap.serializedObject.ApplyModifiedProperties();
            });
            foldOut.Add(mipmapToggle);

            // MipMap container
            var mipMapOptionContainer = new VisualElement();
            mipMapOptionContainer.schedule.Execute(() =>
            {
                var shouldShow = mipmapToggle.value;
                if (mipMapOptionContainer.visible != shouldShow)
                {
                    mipMapOptionContainer.visible = shouldShow;
                    mipMapOptionContainer.EnableInClassList(k_HiddenElementUssClass, !shouldShow);
                }
            }).Every(k_PollForChangesInternal);            
            foldOut.Add(mipMapOptionContainer);
            
            // Border MipMap Toggle
            var borderToggle = new Toggle(styles.borderMipMaps.text)
            {
                value = m_BorderMipMap.intValue > 0,
                tooltip = styles.borderMipMaps.tooltip,
            };
            borderToggle.AddToClassList(k_BaseFieldAlignedUssClass);
            borderToggle.AddToClassList(k_SubElementUssClass);
            borderToggle.RegisterValueChangedCallback(_ =>
            {
                m_BorderMipMap.intValue = borderToggle.value ? 1 : 0;
                m_BorderMipMap.serializedObject.ApplyModifiedProperties();
            });
            mipMapOptionContainer.Add(borderToggle);
            
#if ENABLE_TEXTURE_STREAMING
            // Streaming Toggle
            var streamingToggle = new Toggle(styles.streamingMipMaps.text)
            {
                value = m_StreamingMipmaps.intValue > 0,
                tooltip = styles.streamingMipMaps.tooltip,
            };
            streamingToggle.AddToClassList(k_BaseFieldAlignedUssClass);
            streamingToggle.AddToClassList(k_SubElementUssClass);
            streamingToggle.RegisterValueChangedCallback(_ =>
            {
                m_StreamingMipmaps.intValue = streamingToggle.value ? 1 : 0;
                m_StreamingMipmaps.serializedObject.ApplyModifiedProperties();
            });
            mipMapOptionContainer.Add(streamingToggle);
            
            // Streaming Priority Field
            var streamingPriorityField = new PropertyField(m_StreamingMipmapsPriority, styles.streamingMipmapsPriority.text)
            {
                tooltip = styles.streamingMipmapsPriority.tooltip,
            };
            streamingPriorityField.AddToClassList(k_BaseFieldAlignedUssClass);
            streamingPriorityField.AddToClassList(k_SubSubElementUssClass);
            streamingPriorityField.RegisterValueChangeCallback(x =>
            {
                m_StreamingMipmapsPriority.intValue = Mathf.Clamp(m_StreamingMipmapsPriority.intValue, -128, 127);
                m_StreamingMipmapsPriority.serializedObject.ApplyModifiedProperties();
            });  
            streamingPriorityField.schedule.Execute(() =>
            {
                var shouldShow = streamingToggle.value;
                if (streamingPriorityField.visible != shouldShow)
                {
                    streamingPriorityField.visible = shouldShow;
                    streamingPriorityField.EnableInClassList(k_HiddenElementUssClass, !shouldShow);
                }
            }).Every(k_PollForChangesInternal);              
            mipMapOptionContainer.Add(streamingPriorityField);
#endif
            
            // Mip Map Filtering Enum
            var mipmapModeEnum = new EnumField(styles.mipMapFilter.text, TextureImporterMipFilter.BoxFilter)
            {
                value = (TextureImporterMipFilter)m_MipMapMode.intValue,
                tooltip = styles.mipMapFilter.tooltip
            };
            mipmapModeEnum.AddToClassList(k_BaseFieldAlignedUssClass);
            mipmapModeEnum.AddToClassList(k_SubElementUssClass);
            mipmapModeEnum.RegisterValueChangedCallback(_ =>
            {
                m_MipMapMode.intValue = (int)(TextureImporterMipFilter)mipmapModeEnum.value;
                m_MipMapMode.serializedObject.ApplyModifiedProperties();
            });
            mipMapOptionContainer.Add(mipmapModeEnum);  
            
            // Preserve Coverage Toggle
            var preserveCoverageToggle = new Toggle(styles.mipMapsPreserveCoverage.text)
            {
                value = m_MipMapsPreserveCoverage.intValue > 0,
                tooltip = styles.mipMapsPreserveCoverage.tooltip,
            };
            preserveCoverageToggle.AddToClassList(k_BaseFieldAlignedUssClass);
            preserveCoverageToggle.AddToClassList(k_SubElementUssClass);
            preserveCoverageToggle.RegisterValueChangedCallback(_ =>
            {
                m_MipMapsPreserveCoverage.intValue = preserveCoverageToggle.value ? 1 : 0;
                m_MipMapsPreserveCoverage.serializedObject.ApplyModifiedProperties();
            });
            mipMapOptionContainer.Add(preserveCoverageToggle);
            
            // Alpha Cutoff Field
            var alphaCutoffField = new PropertyField(m_AlphaTestReferenceValue, styles.alphaTestReferenceValue.text)
            {
                tooltip = styles.alphaTestReferenceValue.tooltip,
            };
            alphaCutoffField.AddToClassList(k_BaseFieldAlignedUssClass);
            alphaCutoffField.AddToClassList(k_SubSubElementUssClass);
            alphaCutoffField.schedule.Execute(() =>
            {
                var shouldShow = preserveCoverageToggle.value;
                if (alphaCutoffField.visible != shouldShow)
                {
                    alphaCutoffField.visible = shouldShow;
                    alphaCutoffField.EnableInClassList(k_HiddenElementUssClass, !shouldShow);
                }
            }).Every(k_PollForChangesInternal);              
            mipMapOptionContainer.Add(alphaCutoffField);           
            
            // Fade Out Toggle
            var fadeOutToggle = new Toggle(styles.mipmapFadeOutToggle.text)
            {
                value = m_FadeOut.intValue > 0,
                tooltip = styles.mipmapFadeOutToggle.tooltip,
            };
            fadeOutToggle.AddToClassList(k_BaseFieldAlignedUssClass);
            fadeOutToggle.AddToClassList(k_SubElementUssClass);
            fadeOutToggle.RegisterValueChangedCallback(_ =>
            {
                m_FadeOut.intValue = fadeOutToggle.value ? 1 : 0;
                m_FadeOut.serializedObject.ApplyModifiedProperties();
            });
            mipMapOptionContainer.Add(fadeOutToggle);     
            
            // Fade Distance Slider
            var fadeDistanceSlider = new MinMaxSlider(styles.mipmapFadeOut.text, m_MipMapFadeDistanceStart.intValue, m_MipMapFadeDistanceEnd.intValue, 0, 10)
            {
                tooltip = styles.mipmapFadeOut.tooltip
            };
            fadeDistanceSlider.AddToClassList(k_BaseFieldAlignedUssClass);
            fadeDistanceSlider.AddToClassList(k_SubSubElementUssClass);
            fadeDistanceSlider.RegisterValueChangedCallback(_ =>
            {
                m_MipMapFadeDistanceStart.intValue = Mathf.RoundToInt(fadeDistanceSlider.minValue);
                m_MipMapFadeDistanceEnd.intValue = Mathf.RoundToInt(fadeDistanceSlider.maxValue);
                m_MipMapFadeDistanceStart.serializedObject.ApplyModifiedProperties();
                m_MipMapFadeDistanceEnd.serializedObject.ApplyModifiedProperties();
            });
            fadeDistanceSlider.schedule.Execute(() =>
            {
                var shouldShow = fadeOutToggle.value;
                if (fadeDistanceSlider.visible != shouldShow)
                {
                    fadeDistanceSlider.visible = shouldShow;
                    fadeDistanceSlider.EnableInClassList(k_HiddenElementUssClass, !shouldShow);
                }
            }).Every(k_PollForChangesInternal);              
            mipMapOptionContainer.Add(fadeDistanceSlider);              
        }

        void InitPreview()
        {
            if (m_ImporterTargets.Length > 1)
                return;

            var importerPath = m_AssetPaths[0];
            var gameObject = AssetDatabase.LoadAssetAtPath<GameObject>(importerPath);

            if (m_ModelPreviewer != null)
            {
                m_ModelPreviewer.Dispose();
                m_ModelPreviewer = null;
            }

            if (gameObject != null &&
            m_ImporterTargets[0].importMode == FileImportModes.AnimatedSprite &&
            gameObject.GetComponent<Animator>())
            {
                var clips = GetAllAnimationClips(importerPath);
                m_ModelPreviewer = new ModelPreviewer(gameObject, clips);
                m_DefaultClip = clips != null && clips.Length > 0 ? clips[0] : null;
            }
        }

        void DisposePreview()
        {
            if (m_ModelPreviewer == null)
                return;

            m_ModelPreviewer.Dispose();
            m_ModelPreviewer = null;
        }

        static AnimationClip[] GetAllAnimationClips(string assetPath)
        {
            var assets = AssetDatabase.LoadAllAssetsAtPath(assetPath);
            var clips = new List<AnimationClip>();
            for (var i = 0; i < assets.Length; ++i)
            {
                if (assets[i] is AnimationClip clip)
                    clips.Add(clip);
            }

            return clips.ToArray();
        }

        /// <summary>
        /// Override for AssetImporter.extraDataType
        /// </summary>
        protected override Type extraDataType => typeof(AsepriteImporterEditorExternalData);

        /// <summary>
        /// Override for AssetImporter.InitializeExtraDataInstance
        /// </summary>
        /// <param name="extraTarget">Target object</param>
        /// <param name="targetIndex">Target index</param>
        protected override void InitializeExtraDataInstance(UnityEngine.Object extraTarget, int targetIndex)
        {
            var importer = targets[targetIndex] as AsepriteImporter;
            var extraData = extraTarget as AsepriteImporterEditorExternalData;
            var platformSettingsNeeded = TexturePlatformSettingsHelper.PlatformSettingsNeeded(PlatformSettingsUtilities.CreateDefaultSettings);
            if (importer != null)
            {
                extraData.Init(importer, platformSettingsNeeded);
            }
        }

        /// <summary>
        /// Implementation of virtual method CreateInspectorGUI.
        /// </summary>
        /// <returns>VisualElement container for Inspector visual.</returns>
        public override VisualElement CreateInspectorGUI()
        {
            m_RootVisualElement = new VisualElement()
            {
                name = "Root"
            };

            var styleSheet = EditorGUIUtility.Load("packages/com.unity.2d.aseprite/Editor/Assets/UI/AsepriteImporterStyleSheet.uss") as StyleSheet;
            m_RootVisualElement.styleSheets.Add(styleSheet);

            m_InspectorSettingsView = new VisualElement()
            {
                name = "InspectorSettings"
            };
            m_RootVisualElement.Add(m_InspectorSettingsView);

            ShowInspectorTab(m_ActiveEditorIndex);

            return m_RootVisualElement;
        }

        /// <summary>
        /// Implementation of AssetImporterEditor.OnDisable
        /// </summary>
        public override void OnDisable()
        {
            base.OnDisable();
            DisposePreview();

            if (m_RootVisualElement != null)
                m_RootVisualElement.Clear();
        }
        
        /// <summary>
        /// Implementation of AssetImporterEditor.DiscardChanges.
        /// </summary>
        public override void DiscardChanges()
        {
            base.DiscardChanges();
            m_TexturePlatformSettingsHelper = new TexturePlatformSettingsHelper(this);
        }

        void ShowInspectorTab(int tab)
        {
            m_InspectorSettingsView.Clear();
            m_InspectorSettingsView.Add(m_InspectorUI[tab]);
        }
        
        /// <summary>
        /// Implementation of AssetImporterEditor.SaveChanges.
        /// </summary>
        public override void SaveChanges()
        {
            ApplyTexturePlatformSettings();

            serializedObject.ApplyModifiedProperties();
            extraDataSerializedObject.ApplyModifiedProperties();
            base.SaveChanges();
        }

        // showPerAxisWrapModes is state of whether "Per-Axis" mode should be active in the main dropdown.
        // It is set automatically if wrap modes in UVW are different, or if user explicitly picks "Per-Axis" option -- when that one is picked,
        // then it should stay true even if UVW wrap modes will initially be the same.
        //
        // Note: W wrapping mode is only shown when isVolumeTexture is true.
        static void WrapModePopup(SerializedProperty wrapU, SerializedProperty wrapV, SerializedProperty wrapW, bool isVolumeTexture, ref bool showPerAxisWrapModes)
        {
            // In texture importer settings, serialized properties for things like wrap modes can contain -1;
            // that seems to indicate "use defaults, user has not changed them to anything" but not totally sure.
            // Show them as Repeat wrap modes in the popups.
            var wu = (TextureWrapMode)Mathf.Max(wrapU.intValue, 0);
            var wv = (TextureWrapMode)Mathf.Max(wrapV.intValue, 0);
            var ww = (TextureWrapMode)Mathf.Max(wrapW.intValue, 0);

            // automatically go into per-axis mode if values are already different
            if (wu != wv)
                showPerAxisWrapModes = true;
            if (isVolumeTexture)
            {
                if (wu != ww || wv != ww)
                    showPerAxisWrapModes = true;
            }

            // It's not possible to determine whether any single texture in the whole selection is using per-axis wrap modes
            // just from SerializedProperty values. They can only tell if "some values in whole selection are different" (e.g.
            // wrap value on U axis is not the same among all textures), and can return value of "some" object in the selection
            // (typically based on object loading order). So in order for more intuitive behavior with multi-selection,
            // we go over the actual objects when there's >1 object selected and some wrap modes are different.
            if (!showPerAxisWrapModes)
            {
                if (wrapU.hasMultipleDifferentValues || wrapV.hasMultipleDifferentValues || (isVolumeTexture && wrapW.hasMultipleDifferentValues))
                {
                    if (IsAnyTextureObjectUsingPerAxisWrapMode(wrapU.serializedObject.targetObjects, isVolumeTexture))
                    {
                        showPerAxisWrapModes = true;
                    }
                }
            }

            int value = showPerAxisWrapModes ? -1 : (int)wu;

            // main wrap mode popup
            EditorGUI.BeginChangeCheck();
            EditorGUI.showMixedValue = !showPerAxisWrapModes && (wrapU.hasMultipleDifferentValues || wrapV.hasMultipleDifferentValues || (isVolumeTexture && wrapW.hasMultipleDifferentValues));
            value = EditorGUILayout.IntPopup(styles.wrapModeLabel, value, styles.wrapModeContents, styles.wrapModeValues);
            if (EditorGUI.EndChangeCheck() && value != -1)
            {
                // assign the same wrap mode to all axes, and hide per-axis popups
                wrapU.intValue = value;
                wrapV.intValue = value;
                wrapW.intValue = value;
                showPerAxisWrapModes = false;

                wrapU.serializedObject.ApplyModifiedProperties();
                wrapV.serializedObject.ApplyModifiedProperties();
                wrapW.serializedObject.ApplyModifiedProperties();
            }

            // show per-axis popups if needed
            if (value == -1)
            {
                showPerAxisWrapModes = true;
                EditorGUI.indentLevel++;
                WrapModeAxisPopup(styles.wrapU, wrapU);
                WrapModeAxisPopup(styles.wrapV, wrapV);
                if (isVolumeTexture)
                {
                    WrapModeAxisPopup(styles.wrapW, wrapW);
                }
                EditorGUI.indentLevel--;
            }
            EditorGUI.showMixedValue = false;
        }

        static void WrapModeAxisPopup(GUIContent label, SerializedProperty wrapProperty)
        {
            // In texture importer settings, serialized properties for wrap modes can contain -1, which means "use default".
            var wrap = (TextureWrapMode)Mathf.Max(wrapProperty.intValue, 0);
            Rect rect = EditorGUILayout.GetControlRect();
            EditorGUI.BeginChangeCheck();
            EditorGUI.BeginProperty(rect, label, wrapProperty);
            wrap = (TextureWrapMode)EditorGUI.EnumPopup(rect, label, wrap);
            EditorGUI.EndProperty();
            if (EditorGUI.EndChangeCheck())
            {
                wrapProperty.intValue = (int)wrap;
            }
        }

        static bool IsAnyTextureObjectUsingPerAxisWrapMode(UnityEngine.Object[] objects, bool isVolumeTexture)
        {
            foreach (var o in objects)
            {
                int u = 0, v = 0, w = 0;
                // the objects can be Textures themselves, or texture-related importers
                if (o is Texture)
                {
                    var ti = (Texture)o;
                    u = (int)ti.wrapModeU;
                    v = (int)ti.wrapModeV;
                    w = (int)ti.wrapModeW;
                }
                if (o is TextureImporter)
                {
                    var ti = (TextureImporter)o;
                    u = (int)ti.wrapModeU;
                    v = (int)ti.wrapModeV;
                    w = (int)ti.wrapModeW;
                }
                if (o is IHVImageFormatImporter)
                {
                    var ti = (IHVImageFormatImporter)o;
                    u = (int)ti.wrapModeU;
                    v = (int)ti.wrapModeV;
                    w = (int)ti.wrapModeW;
                }
                u = Mathf.Max(0, u);
                v = Mathf.Max(0, v);
                w = Mathf.Max(0, w);
                if (u != v)
                {
                    return true;
                }
                if (isVolumeTexture)
                {
                    if (u != w || v != w)
                    {
                        return true;
                    }
                }
            }
            return false;
        }
        
        /// <summary>
        /// Implementation of AssetImporterEditor.Apply
        /// </summary>
        protected override void Apply()
        {
            m_TexturePlatformSettingsHelper.Apply();
            ApplyTexturePlatformSettings();
            InternalEditorBridge.ApplySpriteEditorWindow();
            base.Apply();

            if (m_ModelPreviewer != null)
            {
                m_ModelPreviewer.Dispose();
                m_ModelPreviewer = null;
            }
        }

        void ApplyTexturePlatformSettings()
        {
            for (var i = 0; i < targets.Length && i < extraDataTargets.Length; ++i)
            {
                var asepriteImporter = (AsepriteImporter)targets[i];
                var externalData = (AsepriteImporterEditorExternalData)extraDataTargets[i];
                foreach (var ps in externalData.platformSettings)
                {
                    asepriteImporter.SetImporterPlatformSettings(ps);
                    asepriteImporter.Apply();
                }
            }
        }

        /// <summary>
        /// Override of AssetImporterEditor.HasModified.
        /// </summary>
        /// <returns>Returns True if has modified data. False otherwise.</returns>
        public override bool HasModified()
        {
            if (base.HasModified())
                return true;

            return m_TexturePlatformSettingsHelper.HasModified();
        }

        /// <summary>
        /// Implementation of ITexturePlatformSettingsDataProvider.GetTargetCount.
        /// </summary>
        /// <returns>Returns the number of selected targets.</returns>
        int ITexturePlatformSettingsDataProvider.GetTargetCount() => targets.Length;

        /// <summary>
        /// ITexturePlatformSettingsDataProvider.GetPlatformTextureSettings.
        /// </summary>
        /// <param name="i">Selected target index.</param>
        /// <param name="name">Name of the platform.</param>
        /// <returns>TextureImporterPlatformSettings for the given platform name and selected target index.</returns>
        TextureImporterPlatformSettings ITexturePlatformSettingsDataProvider.GetPlatformTextureSettings(int i, string name)
        {
            var externalData = extraDataSerializedObject.targetObjects[i] as AsepriteImporterEditorExternalData;
            if (externalData != null)
            {
                foreach (var ps in externalData.platformSettings)
                {
                    if (ps.name == name)
                        return ps;
                }
            }
            return new TextureImporterPlatformSettings()
            {
                name = name,
                overridden = false
            };
        }

        /// <summary>
        /// Implementation of ITexturePlatformSettingsDataProvider.ShowPresetSettings.
        /// </summary>
        /// <returns>True if valid asset is selected, false otherwise.</returns>
        bool ITexturePlatformSettingsDataProvider.ShowPresetSettings()
        {
            return assetTarget == null;
        }

        /// <summary>
        /// Implementation of ITexturePlatformSettingsDataProvider.DoesSourceTextureHaveAlpha.
        /// </summary>
        /// <param name="i">Index to selected target.</param>
        /// <returns>Always returns true since importer deals with source file that has alpha.</returns>
        bool ITexturePlatformSettingsDataProvider.DoesSourceTextureHaveAlpha(int i)
        {
            return true;
        }

        /// <summary>
        /// Implementation of ITexturePlatformSettingsDataProvider.IsSourceTextureHDR.
        /// </summary>
        /// <param name="i">Index to selected target.</param>
        /// <returns>Always returns false since importer does not handle HDR textures.</returns>
        bool ITexturePlatformSettingsDataProvider.IsSourceTextureHDR(int i)
        {
            return false;
        }

        /// <summary>
        /// Implementation of ITexturePlatformSettingsDataProvider.SetPlatformTextureSettings.
        /// </summary>
        /// <param name="i">Selected target index.</param>
        /// <param name="platformSettings">TextureImporterPlatformSettings to apply to target.</param>
        void ITexturePlatformSettingsDataProvider.SetPlatformTextureSettings(int i, TextureImporterPlatformSettings platformSettings)
        {
            var importer = (AsepriteImporter)targets[i];
            var sp = new SerializedObject(importer);
            sp.FindProperty("m_PlatformSettingsDirtyTick").longValue = System.DateTime.Now.Ticks;
            sp.ApplyModifiedProperties();
        }

        /// <summary>
        /// Implementation of ITexturePlatformSettingsDataProvider.GetImporterSettings.
        /// </summary>
        /// <param name="i">Selected target index.</param>
        /// <param name="settings">TextureImporterPlatformSettings reference for data retrieval.</param>
        void ITexturePlatformSettingsDataProvider.GetImporterSettings(int i, TextureImporterSettings settings)
        {
            ((AsepriteImporter)targets[i]).ReadTextureSettings(settings);
            // Get settings that have been changed in the inspector
            GetSerializedPropertySettings(settings);
        }

        /// <summary>
        /// Get the name property of TextureImporterPlatformSettings from a SerializedProperty.
        /// </summary>
        /// <param name="sp">The SerializedProperty to retrive data.</param>
        /// <returns>The name value in string.</returns>
        public string GetBuildTargetName(SerializedProperty sp)
        {
            return sp.FindPropertyRelative("m_Name").stringValue;
        }

        TextureImporterSettings GetSerializedPropertySettings(TextureImporterSettings settings)
        {
            if (!m_AlphaSource.hasMultipleDifferentValues)
                settings.alphaSource = (TextureImporterAlphaSource)m_AlphaSource.intValue;

            if (!m_ConvertToNormalMap.hasMultipleDifferentValues)
                settings.convertToNormalMap = m_ConvertToNormalMap.intValue > 0;

            if (!m_BorderMipMap.hasMultipleDifferentValues)
                settings.borderMipmap = m_BorderMipMap.intValue > 0;

#if ENABLE_TEXTURE_STREAMING
            if (!m_StreamingMipmaps.hasMultipleDifferentValues)
                settings.streamingMipmaps = m_StreamingMipmaps.intValue > 0;
            if (!m_StreamingMipmapsPriority.hasMultipleDifferentValues)
                settings.streamingMipmapsPriority = m_StreamingMipmapsPriority.intValue;
#endif

            if (!m_MipMapsPreserveCoverage.hasMultipleDifferentValues)
                settings.mipMapsPreserveCoverage = m_MipMapsPreserveCoverage.intValue > 0;

            if (!m_AlphaTestReferenceValue.hasMultipleDifferentValues)
                settings.alphaTestReferenceValue = m_AlphaTestReferenceValue.floatValue;

            if (!m_NPOTScale.hasMultipleDifferentValues)
                settings.npotScale = (TextureImporterNPOTScale)m_NPOTScale.intValue;

            if (!m_IsReadable.hasMultipleDifferentValues)
                settings.readable = m_IsReadable.intValue > 0;

            if (!m_sRGBTexture.hasMultipleDifferentValues)
                settings.sRGBTexture = m_sRGBTexture.intValue > 0;

            if (!m_EnableMipMap.hasMultipleDifferentValues)
                settings.mipmapEnabled = m_EnableMipMap.intValue > 0;

            if (!m_MipMapMode.hasMultipleDifferentValues)
                settings.mipmapFilter = (TextureImporterMipFilter)m_MipMapMode.intValue;

            if (!m_FadeOut.hasMultipleDifferentValues)
                settings.fadeOut = m_FadeOut.intValue > 0;

            if (!m_MipMapFadeDistanceStart.hasMultipleDifferentValues)
                settings.mipmapFadeDistanceStart = m_MipMapFadeDistanceStart.intValue;

            if (!m_MipMapFadeDistanceEnd.hasMultipleDifferentValues)
                settings.mipmapFadeDistanceEnd = m_MipMapFadeDistanceEnd.intValue;

            if (!m_SpriteMode.hasMultipleDifferentValues)
                settings.spriteMode = m_SpriteMode.intValue;

            if (!m_SpritePixelsToUnits.hasMultipleDifferentValues)
                settings.spritePixelsPerUnit = m_SpritePixelsToUnits.floatValue;

            if (!m_SpriteExtrude.hasMultipleDifferentValues)
                settings.spriteExtrude = (uint)m_SpriteExtrude.intValue;

            if (!m_SpriteMeshType.hasMultipleDifferentValues)
                settings.spriteMeshType = (SpriteMeshType)m_SpriteMeshType.intValue;

            if (!m_Alignment.hasMultipleDifferentValues)
                settings.spriteAlignment = m_Alignment.intValue;

            if (!m_SpritePivot.hasMultipleDifferentValues)
                settings.spritePivot = m_SpritePivot.vector2Value;

            if (!m_WrapU.hasMultipleDifferentValues)
                settings.wrapModeU = (TextureWrapMode)m_WrapU.intValue;
            if (!m_WrapV.hasMultipleDifferentValues)
                settings.wrapModeU = (TextureWrapMode)m_WrapV.intValue;
            if (!m_WrapW.hasMultipleDifferentValues)
                settings.wrapModeU = (TextureWrapMode)m_WrapW.intValue;

            if (!m_FilterMode.hasMultipleDifferentValues)
                settings.filterMode = (FilterMode)m_FilterMode.intValue;

            if (!m_Aniso.hasMultipleDifferentValues)
                settings.aniso = m_Aniso.intValue;


            if (!m_AlphaIsTransparency.hasMultipleDifferentValues)
                settings.alphaIsTransparency = m_AlphaIsTransparency.intValue > 0;

            if (!m_TextureType.hasMultipleDifferentValues)
                settings.textureType = (TextureImporterType)m_TextureType.intValue;

            if (!m_TextureShape.hasMultipleDifferentValues)
                settings.textureShape = (TextureImporterShape)m_TextureShape.intValue;

            return settings;
        }

        static void ToggleFromInt(SerializedProperty property, GUIContent label)
        {
            EditorGUI.BeginChangeCheck();
            EditorGUI.showMixedValue = property.hasMultipleDifferentValues;
            var value = EditorGUILayout.Toggle(label, property.intValue > 0) ? 1 : 0;
            EditorGUI.showMixedValue = false;
            if (EditorGUI.EndChangeCheck())
                property.intValue = value;
        }

        static void EnumPopup(SerializedProperty property, System.Type type, GUIContent label)
        {
            EditorGUILayout.IntPopup(label.text, property.intValue,
                System.Enum.GetNames(type),
                System.Enum.GetValues(type) as int[]);
        }

        /// <summary>
        /// Override from AssetImporterEditor to show custom preview.
        /// </summary>
        /// <param name="r">Preview Rect.</param>
        public override void DrawPreview(Rect r)
        {
            if (HasRenamedAssets())
                ReloadPreviewData();
            if (m_ModelPreviewer == null)
                InitPreview();

            if (m_ModelPreviewer != null)
                m_ModelPreviewer.DrawPreview(r, "PreBackgroundSolid");
            else
                base.DrawPreview(r);
        }

        void ReloadPreviewData()
        {
            DisposePreview();
            CacheImporterData();
            InitPreview();
            Apply();
        }

        bool HasRenamedAssets()
        {
            for (var i = 0; i < m_AssetPaths.Length; ++i)
            {
                if (m_AssetPaths[i] != (targets[i] as AsepriteImporter).assetPath)
                    return true;
            }
            return false;
        }

        class Styles
        {
            readonly GUIContent textureShape2D = new("2D, Texture is 2D.");
            readonly GUIContent textureShapeCube = new("Cube", "Texture is a Cubemap.");
            public readonly Dictionary<TextureImporterShape, GUIContent[]> textureShapeOptionsDictionnary = new();
            public readonly Dictionary<TextureImporterShape, int[]> textureShapeValuesDictionnary = new();


            public readonly GUIContent filterMode = new("Filter Mode");
            public readonly GUIContent[] filterModeOptions =
            {
                new ("Point (no filter)"),
                new ("Bilinear"),
                new ("Trilinear")
            };

            public readonly GUIContent mipmapFadeOutToggle = new("Fadeout Mip Maps");
            public readonly GUIContent mipmapFadeOut = new("Fade Range");
            public readonly GUIContent readWrite = new("Read/Write Enabled", "Enable to be able to access the raw pixel data from code.");

            public readonly GUIContent alphaSource = new("Alpha Source", "How is the alpha generated for the imported texture.");

            public readonly GUIContent generateMipMaps = new("Generate Mip Maps");
            public readonly GUIContent sRGBTexture = new("sRGB (Color Texture)", "Texture content is stored in gamma space. Non-HDR color textures should enable this flag (except if used for IMGUI).");
            public readonly GUIContent borderMipMaps = new("Border Mip Maps");
#if ENABLE_TEXTURE_STREAMING
            public readonly GUIContent streamingMipMaps = EditorGUIUtility.TrTextContent("Mip Streaming", "Only load larger mipmaps as needed to render the current game cameras. Requires texture streaming to be enabled in quality settings.");
            public readonly GUIContent streamingMipmapsPriority = EditorGUIUtility.TrTextContent("Priority", "Mipmap streaming priority when there's contention for resources. Positive numbers represent higher priority. Valid range is -128 to 127.");
#endif
            public readonly GUIContent mipMapsPreserveCoverage = new("Preserve Coverage", "The alpha channel of generated Mip Maps will preserve coverage during the alpha test.");
            public readonly GUIContent alphaTestReferenceValue = new("Alpha Cutoff Value", "The reference value used during the alpha test. Controls Mip Map coverage.");
            public readonly GUIContent mipMapFilter = new("Mip Map Filtering");

            public readonly List<string> spriteMeshTypeOptions = new()
            {
                L10n.Tr("Full Rect"),
                L10n.Tr("Tight"),
            };

            public readonly GUIContent fileImportMode = new("Import Mode", "How the file should be imported.");
            public readonly GUIContent spritePixelsPerUnit = new("Pixels Per Unit", "How many pixels in the sprite correspond to one unit in the world.");
            public readonly GUIContent spriteMeshType = new("Mesh Type", "Type of sprite mesh to generate.");
            public readonly GUIContent generatePhysicsShape = new("Generate Physics Shape", "Generates a default physics shape from the outline of the Sprite/s when a physics shape has not been set in the Sprite Editor.");

            public readonly GUIContent warpNotSupportWarning = new("Graphics device doesn't support Repeat wrap mode on NPOT textures. Falling back to Clamp.");
            public readonly GUIContent anisoLevelLabel = new("Aniso Level");
            public readonly GUIContent anisotropicDisableInfo = new("Anisotropic filtering is disabled for all textures in Quality Settings.");
            public readonly GUIContent anisotropicForceEnableInfo = new("Anisotropic filtering is enabled for all textures in Quality Settings.");
            public readonly GUIContent unappliedSettingsDialogTitle = new("Unapplied import settings");
            public readonly GUIContent unappliedSettingsDialogContent = new("Unapplied import settings for \'{0}\'.\nApply and continue to sprite editor or cancel.");
            public readonly GUIContent applyButtonLabel = new("Apply");
            public readonly GUIContent cancelButtonLabel = new("Cancel");
            public readonly GUIContent spriteEditorButtonLabel = new("Open Sprite Editor");
            public readonly GUIContent alphaIsTransparency = new("Alpha Is Transparency", "If the provided alpha channel is transparency, enable this to pre-filter the color to avoid texture filtering artifacts. This is not supported for HDR textures.");

            public readonly GUIContent advancedHeaderText = new("Advanced", "Show advanced settings.");

            public readonly GUIContent platformSettingsHeaderText = new GUIContent("Platform Settings");

            public readonly GUIContent[] platformSettingsSelection;

            public readonly GUIContent wrapModeLabel = new("Wrap Mode");
            public readonly GUIContent wrapU = new("U axis");
            public readonly GUIContent wrapV = new("V axis");
            public readonly GUIContent wrapW = new("W axis");


            public readonly GUIContent[] wrapModeContents =
            {
                new ("Repeat"),
                new ("Clamp"),
                new ("Mirror"),
                new ("Mirror Once"),
                new ("Per-axis")
            };
            public readonly int[] wrapModeValues =
            {
                (int)TextureWrapMode.Repeat,
                (int)TextureWrapMode.Clamp,
                (int)TextureWrapMode.Mirror,
                (int)TextureWrapMode.MirrorOnce,
                -1
            };

            public readonly GUIContent importHiddenLayer = EditorGUIUtility.TrTextContent("Include Hidden Layers", "Settings to determine when hidden layers should be imported.");
            public readonly GUIContent defaultPivotSpace = EditorGUIUtility.TrTextContent("Pivot Space", "Select which space the pivot should be calculated in.");
            public readonly GUIContent defaultPivotAlignment = EditorGUIUtility.TrTextContent("Pivot Alignment", "Select where the pivot should be located based on the Pivot Space.");
            public readonly GUIContent customPivotPosition = EditorGUIUtility.TrTextContent("Custom Pivot Position", "Input the normalized position of the Sprite pivots. The position will be calculated based on the Pivot Space.");
            public readonly GUIContent mosaicPadding = EditorGUIUtility.TrTextContent("Mosaic Padding", "External padding between each SpriteRect, in pixels.");
            public readonly GUIContent spritePadding = EditorGUIUtility.TrTextContent("Sprite Padding", "Internal padding within each SpriteRect, in pixels.");

            public readonly List<string> spriteAlignmentOptions = new()
            {
                L10n.Tr("Center"),
                L10n.Tr("Top Left"),
                L10n.Tr("Top"),
                L10n.Tr("Top Right"),
                L10n.Tr("Left"),
                L10n.Tr("Right"),
                L10n.Tr("Bottom Left"),
                L10n.Tr("Bottom"),
                L10n.Tr("Bottom Right"),
                L10n.Tr("Custom"),
            };

            public readonly GUIContent layerImportMode = EditorGUIUtility.TrTextContent("Import Mode", "Choose between generating one Sprite per layer, or merge all layers in a frame into a single Sprite.");
            public readonly List<string> layerImportOptions = new()
            {
                L10n.Tr("Individual Layers"),
                L10n.Tr("Merge Frame")
            };

            public readonly GUIContent generateModelPrefab = EditorGUIUtility.TrTextContent("Model Prefab", "Generate a Model Prefab laid out the same way as inside Aseprite.");

            public readonly GUIContent addSortingGroup = EditorGUIUtility.TrTextContent("Sorting Group", "Add a Sorting Group component to the root of the generated model prefab if it has more than one Sprite Renderer.");
            public readonly GUIContent addShadowCasters = EditorGUIUtility.TrTextContent("Shadow Casters", "Add Shadow Casters on all GameObjects with SpriteRenderer. Note: The Universal Rendering Pipeline package has to be installed.");
            public readonly GUIContent generateAnimationClips = EditorGUIUtility.TrTextContent("Animation Clips", "Generate Animation Clips based on the frame and tag data from the Aseprite file.");
            public readonly GUIContent generateIndividualEvents = EditorGUIUtility.TrTextContent("Individual Events", "Events will be generated with their own method name. If disabled, all events will be received by the method `OnAnimationEvent(string)`.");
            
            public readonly GUIContent generateSpriteAtlas = EditorGUIUtility.TrTextContent("Sprite Atlas", "Generate a Sprite Atlas to contain the created texture. This is to remove any gaps between tiles when drawing a tile map.");

            public readonly GUIContent generalHeaderText = EditorGUIUtility.TrTextContent("General", "General settings.");
            public readonly GUIContent layerImportHeaderText = EditorGUIUtility.TrTextContent("Layer Import", "Layer Import settings.");
            public readonly GUIContent generateAssetsHeaderText = EditorGUIUtility.TrTextContent("Generate Assets", "Generated assets settings.");
            public readonly GUIContent textureHeaderText = EditorGUIUtility.TrTextContent("Texture", "Texture settings.");

            public readonly GUIContent exportAnimationAssetsText = EditorGUIUtility.TrTextContent("Export Animation Assets");
            public readonly GUIContent exportAnimationInfoText = EditorGUIUtility.TrTextContent("To enable the Export Animation Assets button, make sure to first Apply the changes.");

            public Styles()
            {
                // This is far from ideal, but it's better than having tons of logic in the GUI code itself.
                // The combination should not grow too much anyway since only Texture3D will be added later.
                GUIContent[] s2D_Options = { textureShape2D };
                GUIContent[] sCube_Options = { textureShapeCube };
                GUIContent[] s2D_Cube_Options = { textureShape2D, textureShapeCube };
                textureShapeOptionsDictionnary.Add(TextureImporterShape.Texture2D, s2D_Options);
                textureShapeOptionsDictionnary.Add(TextureImporterShape.TextureCube, sCube_Options);
                textureShapeOptionsDictionnary.Add(TextureImporterShape.Texture2D | TextureImporterShape.TextureCube, s2D_Cube_Options);

                int[] s2D_Values = { (int)TextureImporterShape.Texture2D };
                int[] sCube_Values = { (int)TextureImporterShape.TextureCube };
                int[] s2D_Cube_Values = { (int)TextureImporterShape.Texture2D, (int)TextureImporterShape.TextureCube };
                textureShapeValuesDictionnary.Add(TextureImporterShape.Texture2D, s2D_Values);
                textureShapeValuesDictionnary.Add(TextureImporterShape.TextureCube, sCube_Values);
                textureShapeValuesDictionnary.Add(TextureImporterShape.Texture2D | TextureImporterShape.TextureCube, s2D_Cube_Values);

                platformSettingsSelection = new GUIContent[TexturePlatformSettingsModal.validBuildPlatform.Length];
                for (var i = 0; i < TexturePlatformSettingsModal.validBuildPlatform.Length; ++i)
                {
                    platformSettingsSelection[i] = new GUIContent(TexturePlatformSettingsModal.validBuildPlatform[i].buildTargetName);
                }
            }
        }

        static Styles s_Styles;

        static Styles styles
        {
            get
            {
                if (s_Styles == null)
                    s_Styles = new Styles();
                return s_Styles;
            }
        }

        class AsepriteImporterEditorFoldOutState
        {
            readonly SavedBool m_GeneralFoldout;
            readonly SavedBool m_LayerImportFoldout;
            readonly SavedBool m_GenerateAssetFoldout;
            readonly SavedBool m_AdvancedFoldout;
            readonly SavedBool m_TextureFoldout;
            readonly SavedBool m_PlatformSettingsFoldout;

            public bool generalFoldout
            {
                get => m_GeneralFoldout.value;
                set => m_GeneralFoldout.value = value;
            }

            public bool layerImportFoldout
            {
                get => m_LayerImportFoldout.value;
                set => m_LayerImportFoldout.value = value;
            }

            public bool generateAssetFoldout
            {
                get => m_GenerateAssetFoldout.value;
                set => m_GenerateAssetFoldout.value = value;
            }

            public bool advancedFoldout
            {
                get => m_AdvancedFoldout.value;
                set => m_AdvancedFoldout.value = value;
            }

            public bool textureFoldout
            {
                get => m_TextureFoldout.value;
                set => m_TextureFoldout.value = value;
            }

            public bool platformSettingsFoldout
            {
                get => m_PlatformSettingsFoldout.value;
                set => m_PlatformSettingsFoldout.value = value;
            }

            public AsepriteImporterEditorFoldOutState()
            {
                m_GeneralFoldout = new SavedBool("AsepriteImporterEditor.m_GeneralFoldout", true);
                m_LayerImportFoldout = new SavedBool("AsepriteImporterEditor.m_LayerImportFoldout", true);
                m_GenerateAssetFoldout = new SavedBool("AsepriteImporterEditor.m_ExportAssetFoldout", true);
                m_AdvancedFoldout = new SavedBool("AsepriteImporterEditor.m_AdvancedFoldout", false);
                m_TextureFoldout = new SavedBool("AsepriteImporterEditor.m_TextureFoldout", false);
                m_PlatformSettingsFoldout = new SavedBool("AsepriteImporterEditor.m_PlatformSettingsFoldout", false);
            }

            class SavedBool
            {
                readonly string m_Name;
                bool m_Value;
                bool m_Loaded;

                public SavedBool(string name, bool value)
                {
                    m_Name = name;
                    m_Loaded = false;
                    m_Value = value;
                }

                void Load()
                {
                    if (m_Loaded)
                        return;

                    m_Loaded = true;
                    m_Value = EditorPrefs.GetBool(m_Name, m_Value);
                }

                public bool value
                {
                    get
                    {
                        Load();
                        return m_Value;
                    }
                    set
                    {
                        Load();
                        if (m_Value == value)
                            return;
                        m_Value = value;
                        EditorPrefs.SetBool(m_Name, value);
                    }
                }

                public static implicit operator bool(SavedBool s)
                {
                    return s.value;
                }
            }
        }
    }
}
