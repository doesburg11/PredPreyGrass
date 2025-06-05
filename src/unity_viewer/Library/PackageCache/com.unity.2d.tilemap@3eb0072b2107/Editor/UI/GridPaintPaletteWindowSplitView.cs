using UnityEngine;
using UnityEngine.UIElements;

namespace UnityEditor.Tilemaps
{
    internal class GridPaintPaletteWindowSplitView : VisualElement
    {
        private static readonly string ussClassName = "unity-tilepalette-splitview";
        private static readonly string splitViewDataKey = "unity-tilepalette-splitview-data";

        private const float kMinSplitRatio = 0.3f;

        private TwoPaneSplitView m_SplitView;
        private TilePaletteElement m_PaletteElement;
        private TilePaletteBrushElementToggle m_BrushElementToggle;
        private float m_LastSplitRatio = kMinSplitRatio;

        public TilePaletteElement paletteElement => m_PaletteElement;

        public bool isVerticalOrientation
        {
            get
            {
                return m_SplitView.orientation == TwoPaneSplitViewOrientation.Vertical;
            }
            set
            {
                m_SplitView.orientation =
                    value ? TwoPaneSplitViewOrientation.Vertical : TwoPaneSplitViewOrientation.Horizontal;
            }
        }

        private float fullLength => isVerticalOrientation ? layout.height : layout.width;

        private bool isMinSplit => (fullLength - m_SplitView.fixedPaneDimension) <= minBottomSplitDimension;

        private float minTopSplitDimension => isVerticalOrientation ? 24f : 12f;
        private float minBottomSplitDimension => isVerticalOrientation ? 24f : 12f;

        public void ChangeSplitDimensions(float dimension)
        {
            var newLength = fullLength - dimension;
            var diff = newLength - m_SplitView.fixedPaneDimension;
            if (m_SplitView.m_Resizer != null)
                m_SplitView.m_Resizer.ApplyDelta(diff);
        }

        public GridPaintPaletteWindowSplitView(bool isVerticalOrientation)
        {
            AddToClassList(ussClassName);

            name = "tilePaletteSplitView";
            TilePaletteOverlayUtility.SetStyleSheet(this);

            m_PaletteElement = new TilePaletteElement();

            var brushesElement = new TilePaletteBrushModalElement();
            m_SplitView = new TwoPaneSplitView(0, -1, isVerticalOrientation ? TwoPaneSplitViewOrientation.Vertical : TwoPaneSplitViewOrientation.Horizontal);
            m_SplitView.contentContainer.Add(m_PaletteElement);
            m_SplitView.contentContainer.Add(brushesElement);
            Add(m_SplitView);

            m_SplitView.viewDataKey = splitViewDataKey;

            brushesElement.RegisterCallback<GeometryChangedEvent>(OnGeometryChanged);

            m_BrushElementToggle = this.Q<TilePaletteBrushElementToggle>();
            m_BrushElementToggle.ToggleChanged += BrushElementToggleChanged;
            m_BrushElementToggle.SetValueWithoutNotify(!isMinSplit);
        }

        private void BrushElementToggleChanged(bool show)
        {
            var dimension = minBottomSplitDimension;
            if (show)
            {
                dimension = m_LastSplitRatio * fullLength;
                if (dimension < minBottomSplitDimension)
                    dimension = kMinSplitRatio * fullLength;
            }
            ChangeSplitDimensions(dimension);
        }

        private void OnGeometryChanged(GeometryChangedEvent evt)
        {
            m_BrushElementToggle.SetValueWithoutNotify(!isMinSplit);

            if (m_SplitView.fixedPaneDimension < 0f)
            {
                var defaultLength = fullLength * (1.0f - kMinSplitRatio);
                m_SplitView.fixedPaneInitialDimension = defaultLength;
                ChangeSplitDimensions(defaultLength);
            }

            var newDimension = fullLength - m_SplitView.fixedPaneDimension;
            if (fullLength > minBottomSplitDimension)
            {
                // Force the palette toolbar to always be shown
                if (m_SplitView.fixedPaneDimension < minTopSplitDimension)
                {
                    ChangeSplitDimensions(fullLength - minTopSplitDimension);
                }
                // Force the brush toolbar to always be shown
                if (newDimension < minBottomSplitDimension)
                {
                    ChangeSplitDimensions(minBottomSplitDimension);
                }
            }
            if (newDimension > minBottomSplitDimension)
            {
                var newLastSplit = Mathf.Max(newDimension, kMinSplitRatio * fullLength);
                m_LastSplitRatio = newLastSplit / fullLength;
            }
        }
    }
}
