using System;
using UnityEngine;
using UnityEngine.UIElements;

namespace UnityEditor.U2D.Sprites
{
    internal class SpriteFrameCapabilityWindow : EditorWindow
    {
        static long s_LastClosedTime;
        static string s_WarningText = "Editing data from {0} will be deprecated. Unlocking fields for editing is temporary. This will be removed in a future version.";
        Action<EEditCapability, bool> m_OnCapabilityChangedCallback;
        Action<Action, bool> m_OnUndoRedoPerformedCallback;
        EditCapabilityUndoObject m_Capability;
        Toggle m_SpriteNameToggle;
        Toggle m_PositionToggle;
        Toggle m_BorderToggle;
        Toggle m_PivotToggle;
        Toggle m_CreateToggle;
        VisualElement m_Warning;
        TextElement m_WarningText;
        string m_Extension;
        static EEditCapability[] s_LocksToCheck = { EEditCapability.EditSpriteName, EEditCapability.EditSpriteRect, EEditCapability.EditBorder, EEditCapability.EditPivot, EEditCapability.CreateAndDeleteSprite };

        internal static bool ShowAtPosition(Rect buttonRect, EditCapabilityUndoObject sf, string extension, Action<EEditCapability, bool> capabilityChangedCallback, Action<Action, bool> undoCallback)
        {
            long nowMilliSeconds = DateTime.Now.Ticks / TimeSpan.TicksPerMillisecond;
            bool justClosed = nowMilliSeconds < s_LastClosedTime + 50;
            if (!justClosed)
            {
                if (UnityEngine.Event.current != null) // Event.current can be null during integration test
                    UnityEngine.Event.current.Use();

                var window = CreateInstance<SpriteFrameCapabilityWindow>();
                window.Init(buttonRect, sf, extension, capabilityChangedCallback, undoCallback);
                window.ShowPopup();
                return true;
            }
            return false;
        }

        private void UndoRedoPerformed()
        {
            UpdateUI();
        }

        void OnEnable()
        {
            AssemblyReloadEvents.beforeAssemblyReload += Close;
        }

        void OnDisable()
        {
            AssemblyReloadEvents.beforeAssemblyReload -= Close;
            Undo.undoRedoPerformed -= UndoRedoPerformed;
            s_LastClosedTime = DateTime.Now.Ticks / TimeSpan.TicksPerMillisecond;
        }

        private void Init(Rect buttonRect, EditCapabilityUndoObject capability, string extension, Action<EEditCapability, bool> capabilityChangedCallback, Action<Action,bool> undoCallback)
        {
            Undo.undoRedoPerformed += UndoRedoPerformed;
            m_OnCapabilityChangedCallback = capabilityChangedCallback;
            m_Capability = capability;
            m_Extension = extension.ToLower();
            buttonRect = GUIUtility.GUIToScreenRect(buttonRect);
            var windowSize = new Vector2(300, 165);
            if (m_WarningText != null)
                m_WarningText.text = string.Format(s_WarningText, m_Extension);
            UpdateUI();
            ShowAsDropDown(buttonRect, windowSize);
        }

        public void CreateGUI()
        {
            var ve = AssetDatabase.LoadAssetAtPath<VisualTreeAsset>("Packages/com.unity.2d.sprite/Editor/UI/SpriteEditor/SpriteFrameCapabilityWindow.uxml").CloneTree();
            rootVisualElement.Add(ve);
            ve.styleSheets.Add(AssetDatabase.LoadAssetAtPath<StyleSheet>("Packages/com.unity.2d.sprite/Editor/UI/SpriteEditor/SpriteFrameCapabilityWindow.uss"));
            m_SpriteNameToggle = InitToggle(ve, "nameToggle", EEditCapability.EditSpriteName);
            m_PositionToggle = InitToggle(ve, "positionToggle", EEditCapability.EditSpriteRect);
            m_BorderToggle = InitToggle(ve, "borderToggle", EEditCapability.EditBorder);
            m_PivotToggle = InitToggle(ve, "pivotToggle", EEditCapability.EditPivot);
            m_CreateToggle = InitToggle(ve, "createToggle", EEditCapability.CreateAndDeleteSprite);
            m_Warning = ve.Q<VisualElement>("warning");
            m_WarningText = m_Warning.Q<TextElement>("warningText");
            if (m_Extension != null)
                m_WarningText.text = string.Format(s_WarningText, m_Extension);
            UpdateUI();
        }

        Toggle InitToggle(VisualElement element, string name, EEditCapability capability)
        {
            var toggle = element.Q<Toggle>(name);
            toggle.RegisterValueChangedCallback(b => SetCapability(capability, !b.newValue));
            return toggle;
        }

        void SetCapability(EEditCapability capability, bool enable)
        {
            m_OnCapabilityChangedCallback?.Invoke(capability, enable);
            UpdateWarningAndHelp();
        }

        void UpdateUI()
        {
            if (m_Capability != null)
            {
                m_SpriteNameToggle?.SetValueWithoutNotify(!m_Capability.data.HasCapability(EEditCapability.EditSpriteName));
                m_PositionToggle?.SetValueWithoutNotify(!m_Capability.data.HasCapability(EEditCapability.EditSpriteRect));
                m_BorderToggle?.SetValueWithoutNotify(!m_Capability.data.HasCapability(EEditCapability.EditBorder));
                m_PivotToggle?.SetValueWithoutNotify(!m_Capability.data.HasCapability(EEditCapability.EditPivot));
                m_CreateToggle?.SetValueWithoutNotify(!m_Capability.data.HasCapability(EEditCapability.CreateAndDeleteSprite));
                UpdateWarningAndHelp();
            }
        }

        void UpdateWarningAndHelp()
        {
            if (m_Warning != null)
            {
                bool showWarning = false;
                foreach (var @lock in s_LocksToCheck)
                {
                    if(!m_Capability.originalData.HasCapability(@lock) && m_Capability.data.HasCapability(@lock))
                    {
                        showWarning = true;
                        break;
                    }
                }
                m_Warning.style.display = showWarning ? DisplayStyle.Flex : DisplayStyle.None;
            }
        }
    }
}
