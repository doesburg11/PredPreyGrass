using System;
using System.Collections.Generic;
using UnityEditor.U2D;
using UnityEditor.UIElements;
using UnityEngine;
using UnityEngine.U2D;
using UnityEngine.UIElements;

namespace UnityEditor.Tilemaps
{
    internal class TileSetImporterElement : VisualElement
    {
        public Action onRevert;
        public Action onApply;

        private SerializedObject asset;
        private SerializedObject importer;

        private ListView listView;

        private PropertyField cellLayoutField;
        private PropertyField hexagonLayoutField;
        private PropertyField cellSizingField;
        private PropertyField cellSizeField;
        private PropertyField cellGapField;
        private PropertyField sortAxisField;
        private PropertyField sortModeField;

        private PropertyField createAtlasField;

        private VisualElement spriteAtlasEditorRoot;
        private PropertyField scriptablePackerElement;
        private Label spriteAtlasLabel;
        private VisualElement applyRevertHe;
        private VisualElement tracker;

        private SpriteAtlas spriteAtlas;
        private Editor spriteAtlasEditor;

        public TileSetImporterElement(SerializedObject inImporter)
        {
            importer = inImporter;

            var openTilePaletteButton = new Button(OpenTilePaletteWindow);
            openTilePaletteButton.text = "Open Tile Palette Window";
            Add(openTilePaletteButton);

            var gridFoldout = new Foldout();
            gridFoldout.Bind(importer);
            gridFoldout.bindingPath = "paletteGridFoldout";
            gridFoldout.text = "Palette Grid";

            cellLayoutField = new PropertyField()
            {
                bindingPath = "m_CellLayout"
            };

            hexagonLayoutField = new PropertyField
            {
                bindingPath = "m_HexagonLayout"
            };

            cellSizeField = new PropertyField
            {
                bindingPath = "m_CellSize"
            };
            cellGapField = new PropertyField
            {
                bindingPath = "m_CellGap"
            };
            cellSizingField = new PropertyField
            {
                bindingPath = "m_CellSizing"
            };
            sortModeField = new PropertyField
            {
                bindingPath = "m_TransparencySortMode"
            };
            sortAxisField = new PropertyField
            {
                bindingPath = "m_TransparencySortAxis"
            };

            gridFoldout.Add(cellLayoutField);
            gridFoldout.Add(hexagonLayoutField);
            gridFoldout.Add(cellSizingField);
            gridFoldout.Add(cellSizeField);
            gridFoldout.Add(cellGapField);
            gridFoldout.Add(sortModeField);
            gridFoldout.Add(sortAxisField);

            listView = new ListView()
            {
                bindingPath = "m_TextureSources"
                , showAddRemoveFooter = true
                , showBorder = true
                , showAlternatingRowBackgrounds = AlternatingRowBackground.ContentOnly
                , showBoundCollectionSize = false
                , showFoldoutHeader = true
                , reorderable = true
                , reorderMode = ListViewReorderMode.Animated
                , horizontalScrollingEnabled = false
                , headerTitle = "Texture Sources"
                , virtualizationMethod = CollectionVirtualizationMethod.DynamicHeight
            };
            Add(listView);

            var manipulator = new TileSetImporterDragAndDropManipulator( () => true, OnDragPerform);
            var toggle = listView.Q<Toggle>().Q<Label>();
            toggle.AddManipulator(manipulator);

            Add(gridFoldout);

            var textureSourcesFoldout = listView.Q<Foldout>();
            textureSourcesFoldout.Bind(importer);
            textureSourcesFoldout.bindingPath = "textureSourcesFoldout";

            createAtlasField = new PropertyField()
            {
                bindingPath = "m_CreateAtlas"
            };
            Add(createAtlasField);

            spriteAtlasEditorRoot = new VisualElement();
            spriteAtlasEditorRoot.name = "SpriteAtlasEditor";
            Add(spriteAtlasEditorRoot);

            applyRevertHe = new VisualElement();
            applyRevertHe.style.flexDirection = FlexDirection.RowReverse;

            var applyButton = new Button(OnApply);
            applyButton.text = "Apply";

            var revertButton = new Button(OnRevert);
            revertButton.text = "Revert";

            applyRevertHe.Add(applyButton);
            applyRevertHe.Add(revertButton);
            Add(applyRevertHe);

            cellLayoutField.RegisterValueChangeCallback(CellLayoutChanged);
            cellSizingField.RegisterValueChangeCallback(CellSizingChanged);
            sortModeField.RegisterValueChangeCallback(SortModeChanged);
            createAtlasField.RegisterValueChangeCallback(OnAtlasChanged);

            RegisterCallback<AttachToPanelEvent>(OnAttachedToPanel);
            RegisterCallback<DetachFromPanelEvent>(OnDetachedFromPanel);
        }

        private void OnDragPerform(IEnumerable<Texture2D> textures, bool isAlt)
        {
            if (asset == null || textures == null)
                return;

            asset.Update();
            var textureSourcesSP = asset.FindProperty("m_TextureSources");
            if (textureSourcesSP == null)
                return;

            foreach (var texture in textures)
            {
                var i = textureSourcesSP.arraySize;
                textureSourcesSP.InsertArrayElementAtIndex(i);
                var textureSourceSP = textureSourcesSP.GetArrayElementAtIndex(i);
                var textureSP = textureSourceSP.FindPropertyRelative("m_Texture");
                textureSP.objectReferenceValue = texture;
            }
            asset.ApplyModifiedProperties();
        }

        public void Bind(SerializedObject inAsset, SpriteAtlas inAtlas)
        {
            asset = inAsset;
            spriteAtlas = inAtlas;

            cellLayoutField.Bind(asset);
            hexagonLayoutField.Bind(asset);
            cellSizeField.Bind(asset);
            cellGapField.Bind(asset);
            cellSizingField.Bind(asset);
            sortModeField.Bind(asset);
            sortAxisField.Bind(asset);
            createAtlasField.Bind(asset);

            listView.Bind(asset);

            if (RemoveSpriteAtlasEditor())
                AddSpriteAtlasEditor();

            applyRevertHe.SetEnabled(false);

            if (tracker != null)
                Remove(tracker);
            tracker = new VisualElement();
            Add(tracker);
            tracker.TrackSerializedObjectValue(asset, OnSerializedObjectChanged);
        }

        private void OnSerializedObjectChanged(SerializedObject serializedObject)
        {
            applyRevertHe.SetEnabled(true);
        }

        private void OnAttachedToPanel(AttachToPanelEvent evt)
        {
            AddSpriteAtlasEditor();
        }

        private void OnDetachedFromPanel(DetachFromPanelEvent evt)
        {
            RemoveSpriteAtlasEditor();
        }

        private void AddSpriteAtlasEditor()
        {
            if (spriteAtlas == null)
                return;

            Editor.CreateCachedEditor(spriteAtlas, typeof(SpriteAtlasInspector), ref spriteAtlasEditor);
            var spriteAtlasEditorElement = spriteAtlasEditor.CreateInspectorGUI();
            if (spriteAtlasEditorElement == null)
            {
                spriteAtlasEditorElement = new IMGUIContainer(SpriteAtlasEditor_OnGUI);
            }
            if (spriteAtlasEditorRoot != null)
            {
                var foldout = new Foldout();
                foldout.Bind(importer);
                foldout.bindingPath = "spriteAtlasSettingsFoldout";
                foldout.text = "Sprite Atlas Settings";
                foldout.Add(spriteAtlasEditorElement);

                scriptablePackerElement = new PropertyField()
                {
                    bindingPath = "m_ScriptablePacker"
                };
                scriptablePackerElement.Bind(asset);
                scriptablePackerElement.style.marginTop = 3;
                scriptablePackerElement.style.marginBottom = 3;
                foldout.Add(scriptablePackerElement);

                spriteAtlasEditorRoot.Add(foldout);
            }
            spriteAtlasEditorElement.TrackSerializedObjectValue(spriteAtlasEditor.serializedObject, OnSerializedObjectChanged);
        }

        private void SpriteAtlasEditor_OnGUI()
        {
            if (spriteAtlasEditor == null)
                return;

            if (spriteAtlasLabel == null)
            {
                spriteAtlasLabel = scriptablePackerElement.Q<Label>();
            }

            if (spriteAtlasLabel == null)
                return;

            var oldWidth = EditorGUIUtility.labelWidth;
            EditorGUIUtility.labelWidth = spriteAtlasLabel.resolvedStyle.width;
            spriteAtlasEditor.OnInspectorGUI();
            EditorGUIUtility.labelWidth = oldWidth;
        }

        private bool RemoveSpriteAtlasEditor()
        {
            if (spriteAtlasEditorRoot == null)
                return false;

            if (spriteAtlasEditorRoot.childCount == 0)
                return false;

            while (spriteAtlasEditorRoot.childCount > 0)
                spriteAtlasEditorRoot.RemoveAt(0);
            return true;
        }

        private void CellLayoutChanged(SerializedPropertyChangeEvent evt)
        {
            asset.Update();

            var manualIndex = Array.IndexOf(Enum.GetValues(typeof(GridPalette.CellSizing)), GridPalette.CellSizing.Manual);

            switch (evt.changedProperty.enumValueIndex)
            {
                case (int)GridLayout.CellLayout.Rectangle:
                {
                    hexagonLayoutField.SetEnabled(false);
                    asset.FindProperty("m_HexagonLayout").enumValueIndex = (int)TileSet.HexagonLayout.PointTop;
                    asset.FindProperty("m_CellSizing").enumValueIndex = (int)GridPalette.CellSizing.Automatic;
                    asset.FindProperty("m_CellSize").vector3Value =  new Vector3(1, 1, 0);
                }
                break;
                case (int)GridLayout.CellLayout.Hexagon:
                {
                    hexagonLayoutField.SetEnabled(true);
                    asset.FindProperty("m_CellSizing").enumValueIndex = manualIndex;
                    asset.FindProperty("m_CellSize").vector3Value =  new Vector3(0.8659766f, 1, 0);
                }
                break;
                case (int)GridLayout.CellLayout.Isometric:
                {
                    hexagonLayoutField.SetEnabled(false);
                    asset.FindProperty("m_HexagonLayout").enumValueIndex = (int)TileSet.HexagonLayout.PointTop;
                    asset.FindProperty("m_CellSizing").enumValueIndex = manualIndex;
                    asset.FindProperty("m_CellSize").vector3Value =  new Vector3(1, 0.5f, 1);
                }
                break;
                case (int)GridLayout.CellLayout.IsometricZAsY:
                {
                    hexagonLayoutField.SetEnabled(false);
                    asset.FindProperty("m_HexagonLayout").enumValueIndex = (int)TileSet.HexagonLayout.PointTop;
                    asset.FindProperty("m_CellSizing").enumValueIndex = manualIndex;
                    asset.FindProperty("m_CellSize").vector3Value =  new Vector3(1, 0.5f, 1);
                    asset.FindProperty("m_TransparencySortMode").enumValueIndex = (int)TransparencySortMode.CustomAxis;
                    asset.FindProperty("m_TransparencySortAxis").vector3Value = new Vector3(0f, 1f, -0.26f);
                }
                break;
            }
            asset.ApplyModifiedProperties();
        }

        private void CellSizingChanged(SerializedPropertyChangeEvent evt)
        {
            var isManual = evt.changedProperty.enumValueIndex == (int)GridPalette.CellSizing.Manual;
            cellSizeField.SetEnabled(isManual);
            cellGapField.SetEnabled(isManual);
        }

        private void SortModeChanged(SerializedPropertyChangeEvent evt)
        {
            sortAxisField.SetEnabled(evt.changedProperty.enumValueIndex == (int) TransparencySortMode.CustomAxis);
        }

        private void OpenTilePaletteWindow()
        {
            GridPaintPaletteWindow.OpenTilemapPalette();
        }

        private void OnAtlasChanged(SerializedPropertyChangeEvent evt)
        {
            spriteAtlasEditorRoot.SetEnabled(evt.changedProperty.boolValue);
        }

        private void OnApply()
        {
            applyRevertHe.SetEnabled(false);
            spriteAtlasEditor.serializedObject.ApplyModifiedPropertiesWithoutUndo();
            if (onApply != null)
                onApply();
        }

        private void OnRevert()
        {
            applyRevertHe.SetEnabled(false);
            if (onRevert != null)
                onRevert();
        }
    }
}
