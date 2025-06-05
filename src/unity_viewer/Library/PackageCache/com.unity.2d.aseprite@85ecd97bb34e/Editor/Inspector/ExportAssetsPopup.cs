using UnityEngine;
using UnityEngine.UIElements;

namespace UnityEditor.U2D.Aseprite
{
    internal class ExportAssetsPopup : EditorWindow
    {
        static class Content
        {
            public static readonly string titleText = L10n.Tr("Export Animation Assets");
            public static readonly string bodyText = L10n.Tr("Select which assets should be exported");
            public static readonly string controllerText = L10n.Tr("Animator Controller");
            public static readonly string clipsText = L10n.Tr("Animation Clips");
            public static readonly string exportText = L10n.Tr("Export");
            public static readonly string cancelText = L10n.Tr("Cancel");
        }

        const string k_SubElementUssClass = "SubElement";
        const string k_ControllerToggleId = "ControllerToggle";
        const string k_ClipsToggleId = "ClipsToggle";

        AsepriteImporterEditor m_ImporterEditor;
        AsepriteImporter[] m_ImporterTargets;

        Toggle m_ControllerToggle;
        Toggle m_ClipsToggle;


        void Awake()
        {
            titleContent = new GUIContent(Content.titleText);

            var size = new Vector2(300f, 110f);
            maxSize = size;
            minSize = size;
        }

        void OnSelectionChange()
        {
            Close();
        }

        public void ShowExportPopup(AsepriteImporterEditor importerEditor, AsepriteImporter[] importerTargets)
        {
            m_ImporterEditor = importerEditor;
            m_ImporterTargets = importerTargets;
            ShowPopup();
        }

        void CreateGUI()
        {
            var styleSheet = EditorGUIUtility.Load("packages/com.unity.2d.aseprite/Editor/Assets/UI/ExportAnimAssetsStyleSheet.uss") as StyleSheet;
            rootVisualElement.styleSheets.Add(styleSheet);

            var headerLabel = new Label(Content.bodyText)
            {
                name = "Header"
            };
            rootVisualElement.Add(headerLabel);

            m_ControllerToggle = new Toggle(Content.controllerText)
            {
                value = GetEditorPrefsBool(k_ControllerToggleId, true)
            };
            m_ControllerToggle.RegisterValueChangedCallback(x =>
            {
                SetEditorPrefsBool(k_ControllerToggleId, m_ControllerToggle.value);
            });
            m_ControllerToggle.AddToClassList(k_SubElementUssClass);
            rootVisualElement.Add(m_ControllerToggle);

            m_ClipsToggle = new Toggle(Content.clipsText)
            {
                value = GetEditorPrefsBool(k_ClipsToggleId, false)
            };
            m_ClipsToggle.RegisterValueChangedCallback(x =>
            {
                SetEditorPrefsBool(k_ClipsToggleId, m_ClipsToggle.value);
            });
            m_ClipsToggle.AddToClassList(k_SubElementUssClass);
            rootVisualElement.Add(m_ClipsToggle);

            var buttonArea = new VisualElement()
            {
                name = "ButtonArea"
            };
            rootVisualElement.Add(buttonArea);

            var cancelButton = new Button()
            {
                text = Content.cancelText
            };
            cancelButton.clicked += Close;
            buttonArea.Add(cancelButton);

            var exportButton = new Button()
            {
                text = Content.exportText
            };
            exportButton.clicked += () =>
            {
                var exportClips = m_ClipsToggle.value;
                var exportController = m_ControllerToggle.value;
                ImportUtilities.ExportAnimationAssets(m_ImporterTargets, exportClips, exportController);
                m_ImporterEditor.SaveChanges();
                Close();
            };
            buttonArea.Add(exportButton);
        }

        bool GetEditorPrefsBool(string id, bool defaultValue) => EditorPrefs.GetBool(GetType().Name + id, defaultValue);
        void SetEditorPrefsBool(string id, bool newValue) => EditorPrefs.SetBool(GetType().Name + id, newValue);
    }
}
