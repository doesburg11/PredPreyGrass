using System;
using System.IO;
using UnityEditor.U2D.Common;
using UnityEditor.ShortcutManagement;
using UnityEngine;
using UnityEngine.U2D.Animation;
using UnityEngine.UIElements;

namespace UnityEditor.U2D.Animation.SpriteLibraryEditor
{
    internal interface ISpriteLibraryEditorWindow
    {
        void SaveChanges();
        void DiscardChanges();

        bool hasUnsavedChanges { get; }
        void HandleUnsavedChanges();
        void HandleRevertChanges();
    }

    internal class SpriteLibraryEditorWindow : EditorWindow, ISpriteLibraryEditorWindow
    {
        public const string tabHeaderName = "EditorTabHeader";

        public const string editorWindowClassName = "sprite-library-editor-window";
        public const string overrideClassName = editorWindowClassName + "__override";
        public const string infoLabelClassName = editorWindowClassName + "__description-text";

        public const string deleteCommandName = "Delete";
        public const string softDeleteCommandName = "SoftDelete";
        public const string renameCommandName = "Rename";

        const string k_WindowTitle = "Sprite Library Editor";
        const string k_AssetNotSelectedLabelClassName = editorWindowClassName + "__asset-not-selected-text";

        EditorTopToolbar m_TopToolbar;
        EditorBottomToolbar m_BottomToolbar;
        EditorMainWindow m_MainWindow;

        WindowController m_Controller;

        VisualElement m_EditorWindowRoot;
        VisualElement m_CreateAssetElement;
        ControllerEvents m_ControllerEvents;
        ViewEvents m_ViewEvents;

        InternalEditorBridge.EditorLockTracker m_LockTracker = new InternalEditorBridge.EditorLockTracker();
        GUIStyle m_LockButtonStyle;

        const int k_MinWidth = 500;
        const int k_MinHeight = 300;

        [MenuItem("Window/2D/Sprite Library Editor")]
        public static SpriteLibraryEditorWindow OpenWindow()
        {
            var window = GetWindow<SpriteLibraryEditorWindow>();
            window.m_Controller.SelectAsset(SpriteLibrarySourceAssetImporter.GetAssetFromSelection());
            return window;
        }

        public static SpriteLibraryEditorWindow OpenWindowForAsset(SpriteLibraryAsset asset = null)
        {
            var window = GetWindow<SpriteLibraryEditorWindow>();
            if (asset == null)
                asset = SpriteLibrarySourceAssetImporter.GetAssetFromSelection();
            window.m_Controller.SelectAsset(asset);

            return window;
        }

        [Shortcut("2D/Animation/Save Sprite Library", typeof(SpriteLibraryEditorWindow), KeyCode.S, ShortcutModifiers.Action)]
        public static void SaveShortcut()
        {
            if (focusedWindow is SpriteLibraryEditorWindow spriteLibraryEditorWindow)
                spriteLibraryEditorWindow.m_ViewEvents?.onSave?.Invoke();
        }

        void CreateGUI()
        {
            InitializeWindow();
        }

        void OnDestroy()
        {
            m_Controller.Destroy();

            EditorApplication.playModeStateChanged -= PlayModeStateChanged;
        }

        void InitializeWindow()
        {
            titleContent = new GUIContent(k_WindowTitle, EditorIconUtility.LoadIconResource("Animation.SpriteLibraryManager", "ComponentIcons", "ComponentIcons"));
            saveChangesMessage = TextContent.savePopupMessage;

            m_ControllerEvents = new ControllerEvents();
            m_ViewEvents = new ViewEvents();
            var model = CreateInstance<SpriteLibraryEditorModel>();
            model.hideFlags = HideFlags.HideAndDontSave;

            minSize = new Vector2(k_MinWidth, k_MinHeight);

            m_Controller = new WindowController(this, model, m_ControllerEvents, m_ViewEvents);

            var uiAsset = ResourceLoader.Load<VisualTreeAsset>("SpriteLibraryEditor/SpriteLibraryEditorWindow.uxml");

            var ui = uiAsset.CloneTree();
            if (EditorGUIUtility.isProSkin)
                ui.AddToClassList("Dark");
            ui.StretchToParentSize();

            m_EditorWindowRoot = ui.Q("EditorWindowRoot");

            m_CreateAssetElement = new VisualElement { name = "CreateAssetParent" };
            var descriptionLabel = new Label { name = "DescriptionLabel", text = TextContent.spriteLibraryNoAssetSelected };
            descriptionLabel.AddToClassList(k_AssetNotSelectedLabelClassName);
            m_CreateAssetElement.Add(descriptionLabel);
            var createNewButton = new Button { name = "CreateNewAssetButton", text = TextContent.spriteLibraryCreateNewAsset };
            createNewButton.clicked += HandleCreateNewAsset;
            m_CreateAssetElement.Add(createNewButton);
            ui.Add(m_CreateAssetElement);

            rootVisualElement.Add(ui);

            m_MainWindow = ui.Q<EditorMainWindow>();
            m_MainWindow.BindElements(m_ControllerEvents, m_ViewEvents);

            m_TopToolbar = ui.Q<EditorTopToolbar>();
            m_TopToolbar.BindElements(m_ControllerEvents, m_ViewEvents);

            m_BottomToolbar = ui.Q<EditorBottomToolbar>();
            m_BottomToolbar.BindElements(m_ControllerEvents, m_ViewEvents);

            m_ControllerEvents.onSelectedLibrary.AddListener(_ => UpdateVisualsAfterChange(false));
            m_ControllerEvents.onLibraryDataChanged.AddListener(UpdateVisualsAfterChange);

            UpdateVisualsAfterChange(false);

            HandleEditorPrefs();

            var currentSelection = SpriteLibrarySourceAssetImporter.GetAssetFromSelection();
            if (!m_LockTracker.IsLocked() && m_Controller.GetSelectedAsset() != currentSelection)
                m_Controller.SelectAsset(currentSelection);

            EditorApplication.playModeStateChanged += PlayModeStateChanged;
        }

        void ShowButton(Rect rect)
        {
            m_LockButtonStyle ??= "IN LockButton";
            EditorGUI.BeginChangeCheck();
            m_LockTracker.ShowButtonAtRect(rect, m_LockButtonStyle);
            if (!EditorGUI.EndChangeCheck())
                return;

            m_ViewEvents.onToggleSelectionLock?.Invoke(m_LockTracker.IsLocked());
        }

        void PlayModeStateChanged(PlayModeStateChange newState)
        {
            if (newState == PlayModeStateChange.ExitingEditMode)
            {
                if (hasUnsavedChanges)
                    HandleUnsavedChanges();
            }
        }

        void HandleEditorPrefs()
        {
            m_ViewEvents.onViewSizeUpdate.AddListener(newSize => Settings.viewSize = newSize);
            m_ViewEvents.onViewTypeUpdate.AddListener(viewType => Settings.viewType = viewType);
            m_ViewEvents.onMainUISplitPaneSizeChanged.AddListener(newSize => Settings.splitPaneSize = newSize);
            m_ViewEvents.onToggleAutoSave.AddListener(autoSave => Settings.autoSave = autoSave);
        }

        public override void SaveChanges()
        {
            base.SaveChanges();

            m_Controller.SaveChanges();
        }

        public override void DiscardChanges()
        {
            base.DiscardChanges();

            m_Controller.RevertChanges();
        }

        public void HandleUnsavedChanges()
        {
            if (EditorUtility.DisplayDialog(
                    TextContent.savePopupTitle,
                    TextContent.savePopupMessage,
                    TextContent.savePopupOptionYes,
                    TextContent.savePopupOptionNo))
                SaveChanges();
            else
                DiscardChanges();
        }

        public void HandleRevertChanges()
        {
            if (EditorUtility.DisplayDialog(
                    TextContent.savePopupTitle,
                    TextContent.spriteLibraryRevertMessage,
                    TextContent.savePopupOptionYes,
                    TextContent.savePopupOptionNo))
                m_Controller.RevertChanges();
        }

        void UpdateVisualsAfterChange(bool isModified)
        {
            hasUnsavedChanges = isModified;

            ToggleBetweenMainViewAndEmptyView();
        }

        void ToggleBetweenMainViewAndEmptyView()
        {
            var isEditingAsset = m_Controller.GetSelectedAsset() != null;
            m_EditorWindowRoot.style.display = isEditingAsset ? DisplayStyle.Flex : DisplayStyle.None;
            m_CreateAssetElement.style.display = isEditingAsset ? DisplayStyle.None : DisplayStyle.Flex;
        }

        void HandleCreateNewAsset()
        {
            var newAssetPath = EditorUtility.SaveFilePanelInProject(TextContent.spriteLibraryCreateTitle,
                SpriteLibrarySourceAsset.defaultName, SpriteLibrarySourceAsset.extension.Substring(1),
                TextContent.spriteLibraryCreateMessage);

            if (string.IsNullOrEmpty(newAssetPath) || !string.Equals(Path.GetExtension(newAssetPath), SpriteLibrarySourceAsset.extension, StringComparison.OrdinalIgnoreCase))
                return;

            m_ViewEvents.onCreateNewSpriteLibraryAsset?.Invoke(newAssetPath);
        }

        public static void HandleUnsavedChangesOnApply()
        {
            var windows = Resources.FindObjectsOfTypeAll(typeof(SpriteLibraryEditorWindow));
            var window = windows.Length > 0 ? (SpriteLibraryEditorWindow)windows[0] : null;
            if (window != null)
            {
                if (window.hasUnsavedChanges)
                    window.HandleUnsavedChanges();
            }
        }

        public static void TriggerAssetModifiedOnApply()
        {
            var windows = Resources.FindObjectsOfTypeAll(typeof(SpriteLibraryEditorWindow));
            var window = windows.Length > 0 ? (SpriteLibraryEditorWindow)windows[0] : null;
            if (window != null)
            {
                var controller = window.m_Controller;
                controller.SelectAsset(controller.GetSelectedAsset(), true);
            }
        }

        internal static class Settings
        {
            const string k_SplitPaneSizeKey = UserSettings.kSettingsUniqueKey + "SpriteLibraryEditor.splitPaneSize";
            const string k_AutoSaveKey = UserSettings.kSettingsUniqueKey + "SpriteLibraryEditor.autoSave";
            const string k_ViewTypeKey = UserSettings.kSettingsUniqueKey + "SpriteLibraryEditor.viewType";
            const string k_ViewSizeKey = UserSettings.kSettingsUniqueKey + "SpriteLibraryEditor.viewSize";

            public static float splitPaneSize
            {
                get => EditorPrefs.GetFloat(k_SplitPaneSizeKey, 0.4f);
                set => EditorPrefs.SetFloat(k_SplitPaneSizeKey, Mathf.Clamp(value, 0f, 1f));
            }

            public static bool autoSave
            {
                get => EditorPrefs.GetBool(k_AutoSaveKey, false);
                set => EditorPrefs.SetBool(k_AutoSaveKey, value);
            }

            public static ViewType viewType
            {
                get => (ViewType)EditorPrefs.GetInt(k_ViewTypeKey, 0);
                set => EditorPrefs.SetInt(k_ViewTypeKey, (int)value);
            }

            public static float viewSize
            {
                get => EditorPrefs.GetFloat(k_ViewSizeKey, 0f);
                set => EditorPrefs.SetFloat(k_ViewSizeKey, value);
            }
        }
    }
}