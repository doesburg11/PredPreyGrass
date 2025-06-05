using System;
using UnityEngine;
using UnityEngine.UIElements;

namespace UnityEditor.U2D.Animation.SceneOverlays
{
    internal class OverlayToggle : BaseBoolField
    {
        public OverlayToggle(Texture2D icon, string styleName) : base(null)
        {
            this.Q<VisualElement>(className: inputUssClassName).style.backgroundImage = icon;

            AddToClassList(styleName);
        }
    }

    internal class OverlayToolbar : VisualElement
    {
        static class Styles
        {
            public const string toolbar = SpriteSwapOverlay.rootStyle + "__toolbar";
            public const string thumbnailSettings = SpriteSwapOverlay.rootStyle + "__thumbnail-settings";
            public const string slider = SpriteSwapOverlay.rootStyle + "__slider";
            public const string toggle = SpriteSwapOverlay.rootStyle + "__toggle";
        }

        static class Icons
        {
            public const string filter = "EditorUI.Filter";
            public const string locked = "InspectorLock";
            public const string zoom = "ViewToolZoom";
        }

        public event Action<bool> onFilterToggled;
        public event Action<bool> onLockToggled;
        public event Action onResetSliderValue;
        public event Action<float> onSliderValueChanged;

        public OverlayToolbar()
        {
            AddToClassList(Styles.toolbar);

            var filterIcon = EditorIconUtility.LoadIconResource(Icons.filter, EditorIconUtility.LightIconPath, EditorIconUtility.DarkIconPath);
            var filterToggle = new OverlayToggle(filterIcon, Styles.toggle) { tooltip = TextContent.spriteSwapFilterDescription, value = SpriteSwapOverlay.Settings.filter };
            filterToggle.RegisterValueChangedCallback(evt => onFilterToggled?.Invoke(evt.newValue));
            Add(filterToggle);

            var lockIcon = (Texture2D)EditorGUIUtility.IconContent(Icons.locked).image;
            var lockToggle = new OverlayToggle(lockIcon, Styles.toggle) { tooltip = TextContent.spriteSwapLockDescription, value = SpriteSwapOverlay.Settings.locked };
            lockToggle.RegisterValueChangedCallback(evt => onLockToggled?.Invoke(evt.newValue));
            Add(lockToggle);

            var thumbnailSettings = new VisualElement();
            var slider = new Slider { tooltip = TextContent.spriteSwapThumbnailSlider, lowValue = SpriteSwapOverlay.Settings.minThumbnailSize, highValue = SpriteSwapOverlay.Settings.maxThumbnailSize };
            slider.SetValueWithoutNotify(SpriteSwapOverlay.Settings.thumbnailSize);
            slider.RegisterValueChangedCallback(OnSliderValueChanged);
            slider.AddToClassList(Styles.slider);
            var resetButton = new Button { tooltip = TextContent.spriteSwapResetThumbnailSize, style = { minHeight = 18 } };
            var resetImage = new Image { image = (Texture2D)EditorGUIUtility.IconContent(Icons.zoom).image };
            resetButton.Add(resetImage);
            resetButton.clicked += () => OnResetSliderValue(slider);
            resetButton.AddToClassList(Styles.toggle);
            thumbnailSettings.Add(resetButton);
            thumbnailSettings.Add(slider);
            thumbnailSettings.AddToClassList(Styles.thumbnailSettings);
            Add(thumbnailSettings);
        }

        void OnResetSliderValue(Slider slider)
        {
            onResetSliderValue?.Invoke();
            slider.SetValueWithoutNotify(SpriteSwapOverlay.Settings.thumbnailSize);
        }

        void OnSliderValueChanged(ChangeEvent<float> evt)
        {
            onSliderValueChanged?.Invoke(evt.newValue);
        }
    }
}
