using UnityEngine;
using UnityEditor.U2D.Common;
using UnityEngine.UIElements;
using UnityEditor.UIElements;
using System.Collections.Generic;
using Object = UnityEngine.Object;

namespace UnityEditor.U2D.Animation.Upgrading
{
    internal class AssetUpgraderWindow : EditorWindow
    {
        static class ElementIds
        {
            public const string ModeSelector = "ModeSelector";
            public const string ToolDescription = "ToolDescription";
            public const string WarningImage = "WarningImage";
            public const string WarningCount = "WarningCount";
            public const string ErrorImage = "ErrorImage";
            public const string ErrorCount = "ErrorCount";
            public const string SuccessImage = "SuccessImage";
            public const string SuccessCount = "SuccessCount";
            public const string CenterInfo = "CenterInfo";
            public const string ListHeader = "ListHeader";
            public const string SelectAll = "SelectAll";
            public const string AssetHeader = "AssetHeader";
            public const string AssetList = "AssetList";
            public const string ListFooter = "ListFooter";
            public const string InfoContainer = "InfoContainer";
            public const string InfoBox = "InfoBox";
            public const string Scan = "Scan";
            public const string UpgradeSelected = "UpgradeSelected";
            public const string OpenLog = "OpenLog";
            public const string AssetRow = "AssetRow";
            public const string ObjectToggle = "ObjectToggle";
            public const string ObjectImage = "ObjectImage";
            public const string AssetCheckbox = "AssetCheckbox";
            public const string AssetElement = "AssetElement";
            public const string DarkArea = "DarkArea";
            public const string AssetImage = "AssetImage";
        }

        static class Contents
        {
            public static readonly string WindowTitle = L10n.Tr("2D Anim Asset Upgrader");
            public static readonly string UnsupportedSerializeMode = L10n.Tr("The project's serialization mode is set to {0}. This upgrading tool only supports ForceText. Update the project's serialization mode under `Edit > Project Settings > Editor > Asset Serialization` to use the upgrading tool.");
            public static readonly string SpriteLibDesc = L10n.Tr("Use this tool to replace runtime Sprite Library Assets (.asset) with Sprite Library Source Assets (.spriteLib) for increased tooling support. Once replaced, this tool also makes sure all Sprite Library references in Scenes and Prefabs are maintained.");
            public static readonly string AnimClipDesc = L10n.Tr("Use this tool to upgrade animation clips with older Sprite Resolver data format to the latest Sprite Resolver data format.");
            public static readonly string ScanToBegin = L10n.Tr("Press Scan Project to see which assets require upgrading.");
            public static readonly string NoAssetsRequireUpgrade = L10n.Tr("No assets require upgrading.");
            public static readonly string UpgradeDialogTitle = L10n.Tr("Upgrade Assets");
            public static readonly string UpgradeDialogMessage = L10n.Tr("The upgrade cannot be reverted and AssetBundles would have to be rebuilt but you will get to enjoy the new improvements.\n\nAlternatively, you may choose not to upgrade for now and continue to use the existing assets.\n\nAre you sure you want to upgrade the selected assets?");
            public static readonly string UpgradeDialogYes = L10n.Tr("Yes");
            public static readonly string UpgradeDialogNo = L10n.Tr("No");
            public static readonly string SpriteLibraries = L10n.Tr("Sprite Libraries");
            public static readonly string AnimationClips = L10n.Tr("Animation Clips");
        }

        const string k_UiUxml = "AssetUpgrader/AssetUpgrader.uxml";
        const string k_UiUss = "AssetUpgrader/AssetUpgrader.uss";
        const string k_IconWarningId = "Warning";
        const string k_IconFailId = "False";
        const string k_IconSuccessId = "True";

        Texture2D m_IconWarn;
        Texture2D m_IconFail;
        Texture2D m_IconSuccess;

        ButtonStripField m_ModeSelector;
        Label m_DescriptionLabel;
        Label m_WarningCountLabel;
        Label m_ErrorCountLabel;
        Label m_SuccessCountLabel;
        Label m_CenterInfo;
        VisualElement m_ListHeaderContainer;
        Toggle m_SelectAllToggle;
        Label m_AssetHeader;
        ListView m_AssetListView;
        VisualElement m_ListFooterContainer;
        VisualElement m_InfoContainer;
        HelpBox m_InfoBox;
        Button m_UpgradeSelectedBtn;
        Button m_OpenLogBtn;

        UpgradeMode m_SelectedMode = UpgradeMode.SpriteLibrary;
        List<Object> m_AssetList = new List<Object>();
        List<UpgradeResult> m_UpgradeResultList = new List<UpgradeResult>();
        List<string> m_UpgradeTooltips = new List<string>();
        HashSet<int> m_SelectedObjs = new HashSet<int>();
        string m_UpgradeLogPath;

        [MenuItem("Window/2D/2D Animation Asset Upgrader")]
        internal static void OpenWindow()
        {
            var window = GetWindowWithRect<AssetUpgraderWindow>(new Rect(0, 0, 532, 512));
            window.titleContent = new GUIContent(Contents.WindowTitle);
            window.Show();
        }

        public void CreateGUI()
        {
            var treeAsset = ResourceLoader.Load<VisualTreeAsset>(k_UiUxml);
            rootVisualElement.Add(treeAsset.CloneTree());
            rootVisualElement.styleSheets.Add(ResourceLoader.Load<StyleSheet>(k_UiUss));

            LoadIconTextures();
            SetupUI();
            OnChangeUpgradeMode();
        }

        void LoadIconTextures()
        {
            m_IconWarn = EditorIconUtility.LoadIconResource(k_IconWarningId, EditorIconUtility.LightIconPath, EditorIconUtility.DarkIconPath);
            m_IconFail = EditorIconUtility.LoadIconResource(k_IconFailId, EditorIconUtility.LightIconPath, EditorIconUtility.DarkIconPath);
            m_IconSuccess = EditorIconUtility.LoadIconResource(k_IconSuccessId, EditorIconUtility.LightIconPath, EditorIconUtility.DarkIconPath);
        }

        void SetupUI()
        {
            SetupTopContainer();
            SetupConversionResultContainer();
            SetupCenterContainer();
            SetupInfoContainer();
            SetupBottomContainer();
        }

        void SetupTopContainer()
        {
            m_ModeSelector = rootVisualElement.Q<ButtonStripField>(ElementIds.ModeSelector);
            m_ModeSelector.focusable = false;
            m_ModeSelector.AddButton(Contents.SpriteLibraries);
            m_ModeSelector.AddButton(Contents.AnimationClips);

            m_ModeSelector.SetValueWithoutNotify((int)m_SelectedMode);
            m_ModeSelector.RegisterValueChangedCallback(_ => OnChangeUpgradeMode());

            m_DescriptionLabel = rootVisualElement.Q<Label>(ElementIds.ToolDescription);
        }

        void SetupConversionResultContainer()
        {
            var warningImage = rootVisualElement.Q<Image>(ElementIds.WarningImage);
            warningImage.image = m_IconWarn;
            m_WarningCountLabel = rootVisualElement.Q<Label>(ElementIds.WarningCount);

            var errorImage = rootVisualElement.Q<Image>(ElementIds.ErrorImage);
            errorImage.image = m_IconFail;
            m_ErrorCountLabel = rootVisualElement.Q<Label>(ElementIds.ErrorCount);

            var successImage = rootVisualElement.Q<Image>(ElementIds.SuccessImage);
            successImage.image = m_IconSuccess;
            m_SuccessCountLabel = rootVisualElement.Q<Label>(ElementIds.SuccessCount);
        }

        void SetupCenterContainer()
        {
            m_CenterInfo = rootVisualElement.Q<Label>(ElementIds.CenterInfo);

            m_ListHeaderContainer = rootVisualElement.Q<VisualElement>(ElementIds.ListHeader);
            m_SelectAllToggle = rootVisualElement.Q<Toggle>(ElementIds.SelectAll);
            m_SelectAllToggle.RegisterCallback<ChangeEvent<bool>>(OnSelectAll);
            m_AssetHeader = rootVisualElement.Q<Label>(ElementIds.AssetHeader);
            m_AssetHeader.SetEnabled(false);

            m_AssetListView = rootVisualElement.Q<ListView>(ElementIds.AssetList);
            m_AssetListView.itemsSource = m_AssetList;
            m_AssetListView.Rebuild();

            m_ListFooterContainer = rootVisualElement.Q<VisualElement>(ElementIds.ListFooter);
        }

        void SetupInfoContainer()
        {
            m_InfoContainer = rootVisualElement.Q<VisualElement>(ElementIds.InfoContainer);
            m_InfoBox = rootVisualElement.Q<HelpBox>(ElementIds.InfoBox);
        }

        void SetupBottomContainer()
        {
            var scanBtn = rootVisualElement.Q<Button>(ElementIds.Scan);
            scanBtn.clicked += OnScanClicked;
            scanBtn.SetEnabled(CanUseTool());

            m_UpgradeSelectedBtn = rootVisualElement.Q<Button>(ElementIds.UpgradeSelected);
            m_UpgradeSelectedBtn.clicked += OnUpgradeSelectedClicked;

            m_OpenLogBtn = rootVisualElement.Q<Button>(ElementIds.OpenLog);
            m_OpenLogBtn.clicked += OnOpenLogClicked;
        }

        static bool CanUseTool() => EditorSettings.serializationMode == SerializationMode.ForceText;

        void OnChangeUpgradeMode()
        {
            var newMode = (UpgradeMode)m_ModeSelector.value;
            m_SelectedMode = newMode;

            if (m_SelectedMode == UpgradeMode.SpriteLibrary)
                m_AssetHeader.text = Contents.SpriteLibraries;
            else
                m_AssetHeader.text = Contents.AnimationClips;

            m_AssetList.Clear();
            m_UpgradeResultList.Clear();
            m_UpgradeTooltips.Clear();
            m_AssetListView.Rebuild();

            m_CenterInfo.text = Contents.ScanToBegin;
            m_CenterInfo.SetHiddenFromLayout(false);
            m_ListHeaderContainer.SetHiddenFromLayout(true);
            m_SelectAllToggle.SetValueWithoutNotify(false);
            m_AssetListView.SetHiddenFromLayout(true);
            m_ListFooterContainer.SetHiddenFromLayout(true);
            m_UpgradeSelectedBtn.SetEnabled(false);
            m_OpenLogBtn.SetEnabled(false);

            ResetConversionResultCounters();
            ResetInfoBox();

            switch (m_SelectedMode)
            {
                case UpgradeMode.SpriteLibrary:
                    m_DescriptionLabel.text = Contents.SpriteLibDesc;
                    break;
                case UpgradeMode.AnimationClip:
                    m_DescriptionLabel.text = Contents.AnimClipDesc;
                    break;
            }
        }

        void ResetInfoBox()
        {
            if (CanUseTool())
                m_InfoContainer.SetHiddenFromLayout(true);
            else
            {
                m_InfoContainer.SetHiddenFromLayout(false);
                m_InfoBox.text = string.Format(Contents.UnsupportedSerializeMode, EditorSettings.serializationMode.ToString());
                m_InfoBox.messageType = HelpBoxMessageType.Error;
            }
        }

        void OnScanClicked()
        {
            m_AssetList = AssetUpgrader.GetAllAssetsOfType(m_SelectedMode);

            m_UpgradeResultList.Clear();
            m_UpgradeTooltips.Clear();
            for (var i = 0; i < m_AssetList.Count; ++i)
            {
                m_UpgradeTooltips.Add(string.Empty);
                m_UpgradeResultList.Add(UpgradeResult.None);
            }

            ResetConversionResultCounters();

            m_CenterInfo.SetHiddenFromLayout(m_AssetList.Count > 0);
            if (m_CenterInfo.enabledSelf)
                m_CenterInfo.text = Contents.NoAssetsRequireUpgrade;

            m_ListHeaderContainer.SetHiddenFromLayout(m_AssetList.Count == 0);
            m_AssetListView.SetHiddenFromLayout(m_AssetList.Count == 0);
            m_ListFooterContainer.SetHiddenFromLayout(m_AssetList.Count == 0);
            m_InfoContainer.SetHiddenFromLayout(true);
            m_SelectedObjs.Clear();

            m_SelectAllToggle.SetValueWithoutNotify(true);
            for (var i = 0; i < m_AssetList.Count; ++i)
                AddRemoveSelectedObject(i, true);

            m_AssetListView.itemsSource = m_AssetList;
            m_AssetListView.makeItem = () =>
            {
                var container = new VisualElement();
                container.AddToClassList(ElementIds.AssetRow);

                var toggle = new Toggle();
                toggle.name = ElementIds.ObjectToggle;
                toggle.value = false;
                toggle.AddToClassList(ElementIds.AssetCheckbox);
                container.Add(toggle);

                var objField = new ObjectField();
                objField.value = this;
                objField.allowSceneObjects = false;
                objField.AddToClassList(ElementIds.AssetElement);
                container.Add(objField);

                var imgContainer = new VisualElement();
                imgContainer.AddToClassList(ElementIds.DarkArea);
                container.Add(imgContainer);

                var imgField = new Image();
                imgField.name = ElementIds.ObjectImage;
                imgField.AddToClassList(ElementIds.AssetImage);
                imgContainer.Add(imgField);

                return container;
            };
            m_AssetListView.bindItem = (element, i) =>
            {
                if (m_AssetList[i] == null)
                    return;
                var field = element.Q<ObjectField>();
                field.value = m_AssetList[i];

                var toggle = element.Q<Toggle>(ElementIds.ObjectToggle);
                toggle.RegisterCallback<ChangeEvent<bool>, int>(OnToggleObject, i);
                toggle.SetEnabled(m_UpgradeResultList[i] == UpgradeResult.None);

                var isToggled = m_SelectedObjs.Contains(i);
                toggle.SetValueWithoutNotify(isToggled);

                var resultImage = element.Q<Image>(ElementIds.ObjectImage);
                resultImage.image = GetResultImage(m_UpgradeResultList[i]);
                resultImage.tooltip = m_UpgradeTooltips[i];
            };

            m_AssetListView.Rebuild();
        }

        void OnSelectAll(ChangeEvent<bool> value)
        {
            for (var i = 0; i < m_AssetList.Count; ++i)
            {
                if (m_UpgradeResultList[i] == UpgradeResult.None)
                    AddRemoveSelectedObject(i, value.newValue);
            }

            m_AssetListView.RefreshItems();
        }

        void OnToggleObject(ChangeEvent<bool> value, int index)
        {
            AddRemoveSelectedObject(index, value.newValue);
            m_SelectAllToggle.SetValueWithoutNotify(m_SelectedObjs.Count == m_AssetList.Count);
        }

        void AddRemoveSelectedObject(int index, bool shouldAdd)
        {
            if (shouldAdd && !m_SelectedObjs.Contains(index))
                m_SelectedObjs.Add(index);
            else if (!shouldAdd)
                m_SelectedObjs.Remove(index);

            m_UpgradeSelectedBtn.SetEnabled(m_SelectedObjs.Count > 0);
        }

        void OnUpgradeSelectedClicked()
        {
            if (!EditorUtility.DisplayDialog(Contents.UpgradeDialogTitle, Contents.UpgradeDialogMessage, Contents.UpgradeDialogYes, Contents.UpgradeDialogNo))
                return;

            var selectedObjs = new List<ObjectIndexPair>();
            foreach (var index in m_SelectedObjs)
            {
                selectedObjs.Add(new ObjectIndexPair()
                {
                    Target = m_AssetList[index],
                    Index = index
                });
            }

            m_SelectAllToggle.SetValueWithoutNotify(false);
            m_SelectedObjs.Clear();
            m_UpgradeSelectedBtn.SetEnabled(false);

            var report = AssetUpgrader.UpgradeSelection(m_SelectedMode, selectedObjs);
            if (report.UpgradeEntries != null)
            {
                UpdateObjectResults(report);
                UpdateConversionResultCounters();
                UpdateInfoBox(report);
            }

            m_UpgradeLogPath = UpgradeLogWriter.Generate(report.Log);
            m_OpenLogBtn.SetEnabled(!string.IsNullOrEmpty(m_UpgradeLogPath));
            m_AssetListView.RefreshItems();
        }

        void UpdateObjectResults(UpgradeReport report)
        {
            foreach (var entry in report.UpgradeEntries)
            {
                m_UpgradeResultList[entry.Index] = entry.Result;
                m_UpgradeTooltips[entry.Index] = entry.Message;
                m_AssetList[entry.Index] = entry.Target;
            }
        }

        Texture2D GetResultImage(UpgradeResult result)
        {
            switch (result)
            {
                case UpgradeResult.Successful:
                    return m_IconSuccess;
                case UpgradeResult.Warning:
                    return m_IconWarn;
                case UpgradeResult.Error:
                    return m_IconFail;
                default:
                    return null;
            }
        }

        void UpdateConversionResultCounters()
        {
            var noOfWarnings = 0;
            var noOfErrors = 0;
            var noOfSuccesses = 0;
            foreach (var result in m_UpgradeResultList)
            {
                if (result == UpgradeResult.Warning)
                    noOfWarnings++;
                else if (result == UpgradeResult.Error)
                    noOfErrors++;
                else if (result == UpgradeResult.Successful)
                    noOfSuccesses++;
            }

            m_WarningCountLabel.text = noOfWarnings.ToString();
            m_ErrorCountLabel.text = noOfErrors.ToString();
            m_SuccessCountLabel.text = noOfSuccesses.ToString();
        }

        void UpdateInfoBox(UpgradeReport report)
        {
            var successCount = 0;
            var warningCount = 0;
            var errorCount = 0;
            foreach (var entry in report.UpgradeEntries)
            {
                if (entry.Result == UpgradeResult.Successful)
                    successCount++;
                else if (entry.Result == UpgradeResult.Warning)
                    warningCount++;
                else if (entry.Result == UpgradeResult.Error)
                    errorCount++;
            }

            var summary = "";
            if (warningCount > 0)
            {
                if (warningCount == 1)
                    summary = $"{warningCount} asset was upgraded with warnings.";
                else
                    summary = $"{warningCount} assets were upgraded with warnings.";
                summary += " Open the upgrade log to get more information regarding the warnings.";
                m_InfoBox.messageType = HelpBoxMessageType.Warning;
            }
            else if (errorCount > 0)
            {
                if (errorCount == 1)
                    summary = $"{errorCount} asset failed to upgrade.";
                else
                    summary = $"{errorCount} assets failed to upgrade.";
                summary += " Open the upgrade log to get more information regarding the failures.";
                m_InfoBox.messageType = HelpBoxMessageType.Error;
            }
            else if (successCount > 0)
            {
                if (successCount == 1)
                    summary = $"{successCount} asset was upgraded successfully.";
                else
                    summary = $"{successCount} assets were upgraded successfully.";
                m_InfoBox.messageType = HelpBoxMessageType.Info;
            }

            m_InfoBox.text = summary;
            m_InfoContainer.SetHiddenFromLayout(false);
        }

        void ResetConversionResultCounters()
        {
            m_WarningCountLabel.text = "0";
            m_ErrorCountLabel.text = "0";
            m_SuccessCountLabel.text = "0";
        }

        void OnOpenLogClicked()
        {
            if (!string.IsNullOrEmpty(m_UpgradeLogPath))
                EditorUtility.OpenWithDefaultApp(m_UpgradeLogPath);
        }
    }
}