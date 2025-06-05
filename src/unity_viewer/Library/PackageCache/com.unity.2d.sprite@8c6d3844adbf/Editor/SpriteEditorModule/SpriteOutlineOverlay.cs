using System;
using UnityEngine;
using UnityEngine.UIElements;

namespace UnityEditor.U2D.Sprites.SpriteEditorTool
{
    [UxmlElement]
    internal partial class SpriteOutlineToolOverlayPanel : VisualElement
    {
        static readonly string kGenerateSingle = L10n.Tr("Generate");
        static readonly string kGenerateAll = L10n.Tr("Generate All");
        static readonly string kForceGenerate = L10n.Tr("Force Generate All");

        FloatField m_OutlineDetailField;
        IntegerField m_AlphaToleranceField;
        Slider m_OutlineDetailSlider;
        SliderInt m_AlphaToleranceSlider;
        Toggle m_SnapToggle;
        Toggle m_OptimizeOutline;
        Toggle m_ForceGenerate;
        Button m_GenerateButton;
        Button m_CopyButton;
        Button m_PasteButton;

        public event Action<bool> onGenerateOutline = _ => { };
        public event Action<int> onAlphaToleranceChanged = _ => { };
        public event Action<float> onOutlineDetailChanged  = _ => { };
        public event Action onCopy = () => { };
        public event Action onPaste = () => { };
        public event Action onPasteAll = () => { };
        public event Action onPasteAlternate = () => { };
        public event Action onPasteAlternateAll = () => { };

        public bool snapOn
        {
            get { return m_SnapToggle.value; }
            set { m_SnapToggle.SetValueWithoutNotify(value); }
        }

        public float outlineDetail
        {
            get { return m_OutlineDetailField.value; }
            set
            {
                m_OutlineDetailField.SetValueWithoutNotify(value);
                m_OutlineDetailSlider.SetValueWithoutNotify(value);
            }
        }

        public int alphaTolerance
        {
            get { return m_AlphaToleranceField.value; }
            set
            {
                m_AlphaToleranceField.SetValueWithoutNotify(value);
                m_AlphaToleranceSlider.SetValueWithoutNotify(value);
            }
        }

        public bool optimizeOutline
        {
            get => m_OptimizeOutline.value;
            set => m_OptimizeOutline.SetValueWithoutNotify(value);
        }

        public SpriteOutlineToolOverlayPanel()
        {
            styleSheets.Add(AssetDatabase.LoadAssetAtPath<StyleSheet>("Packages/com.unity.2d.sprite/Editor/UI/SpriteEditor/SpriteOutlineToolOverlayPanelStyle.uss"));
            RegisterCallback<MouseDownEvent>((e) => { e.StopPropagation(); });
            RegisterCallback<MouseUpEvent>((e) => { e.StopPropagation(); });
        }


        public void SetPanelMode(bool hasSelection)
        {
            if (hasSelection)
            {
                m_GenerateButton.text = kGenerateSingle;
                m_CopyButton.SetEnabled(true);
                m_PasteButton.SetEnabled(true);
                m_ForceGenerate.SetEnabled(false);
            }
            else
            {
                UpdateGenerateAllButtonText();
                m_CopyButton.SetEnabled(false);
                m_PasteButton.SetEnabled(false);
                m_ForceGenerate.SetEnabled(true);
            }
        }

        void UpdateGenerateAllButtonText()
        {
            m_GenerateButton.text = m_ForceGenerate.value ? kForceGenerate : kGenerateAll;
        }

        void GenerateOutline()
        {
            onGenerateOutline(m_ForceGenerate.value);
        }

        void OnCopy()
        {
            onCopy();
        }

        void OnPaste()
        {
            onPaste();
        }

        void OnPasteAll()
        {
            onPasteAll();
        }

        void OnPasteAlternate()
        {
            onPasteAlternate();
        }

        void OnPasteAlternateAll()
        {
            onPasteAlternateAll();
        }

        void OnForceGenerateChanged(ChangeEvent<bool> _)
        {
            UpdateGenerateAllButtonText();
        }

        private void BindElements(string alternateLabelText)
        {
            m_GenerateButton = this.Q<Button>("GenerateButton");
            m_GenerateButton.clickable.clicked += GenerateOutline;
            m_GenerateButton.AddManipulator(m_GenerateButton.clickable);

            m_CopyButton = this.Q<Button>("CopyButton");
            m_CopyButton.clickable.clicked += OnCopy;
            m_CopyButton.AddManipulator(m_CopyButton.clickable);

            m_PasteButton = this.Q<Button>("PasteButton");
            m_PasteButton.clickable.clicked += OnPaste;
            m_PasteButton.AddManipulator(m_PasteButton.clickable);

            m_ForceGenerate = this.Q<Toggle>("ForceGenerateToggle");
            m_ForceGenerate.RegisterValueChangedCallback(OnForceGenerateChanged);

            var button = this.Q<Button>("PasteAllButton");
            button.clickable.clicked += OnPasteAll;
            button.AddManipulator(button.clickable);

            button = this.Q<Button>("PasteAlternateButton");
            button.clickable.clicked += OnPasteAlternate;
            button.AddManipulator(button.clickable);

            button = this.Q<Button>("PasteAlternateAllButton");
            button.clickable.clicked += OnPasteAlternateAll;
            button.AddManipulator(button.clickable);

            var alternateLabel = this.Q<Label>("PasteAlternateLabel");
            alternateLabel.text = alternateLabelText;

            m_OutlineDetailField = this.Q<FloatField>("OutlineDetailField");
            m_AlphaToleranceField = this.Q<IntegerField>("AlphaToleranceField");
            m_SnapToggle = this.Q<Toggle>("SnapToggle");
            m_OptimizeOutline = this.Q<Toggle>("OptimizeOutlineToggle");

            m_OutlineDetailSlider = this.Q<Slider>("OutlineDetailSlider");
            m_OutlineDetailSlider.RegisterValueChangedCallback((evt) =>
            {
                if (!evt.newValue.Equals(m_OutlineDetailField.value))
                {
                    m_OutlineDetailField.SetValueWithoutNotify(evt.newValue);
                    onOutlineDetailChanged(evt.newValue);
                }

            });
            m_OutlineDetailField.RegisterValueChangedCallback((evt) =>
            {
                var newValue = evt.newValue;
                if (!newValue.Equals(m_OutlineDetailSlider.value))
                {
                    newValue = Math.Min(newValue, m_OutlineDetailSlider.highValue);
                    newValue = Math.Max(newValue, m_OutlineDetailSlider.lowValue);
                    m_OutlineDetailSlider.value = newValue;
                    m_OutlineDetailField.SetValueWithoutNotify(newValue);
                    onOutlineDetailChanged(evt.newValue);
                }
            });

            m_AlphaToleranceSlider = this.Q<SliderInt>("AlphaToleranceSlider");
            m_AlphaToleranceSlider.RegisterValueChangedCallback((evt) =>
            {
                if (!evt.newValue.Equals(m_OutlineDetailField.value))
                {
                    m_AlphaToleranceField.SetValueWithoutNotify(evt.newValue);
                    onAlphaToleranceChanged(evt.newValue);
                }

            });
            m_AlphaToleranceField.RegisterValueChangedCallback((evt) =>
            {
                var newValue = evt.newValue;
                if (!newValue.Equals(m_AlphaToleranceSlider.value))
                {
                    newValue = Math.Min(newValue, m_AlphaToleranceSlider.highValue);
                    newValue = Math.Max(newValue, m_AlphaToleranceSlider.lowValue);
                    m_AlphaToleranceSlider.value = newValue;
                    m_AlphaToleranceField.SetValueWithoutNotify(newValue);
                    onAlphaToleranceChanged(evt.newValue);
                }
            });
        }

        public static SpriteOutlineToolOverlayPanel GenerateFromUXML(string alternateLabelText)
        {
            var visualTree = AssetDatabase.LoadAssetAtPath<VisualTreeAsset>("Packages/com.unity.2d.sprite/Editor/UI/SpriteEditor/SpriteOutlineToolOverlayPanel.uxml");
            var clone = visualTree.CloneTree().Q<SpriteOutlineToolOverlayPanel>("SpriteOutlineToolOverlayPanel");
            clone.BindElements(alternateLabelText);
            return clone;
        }
    }
}
