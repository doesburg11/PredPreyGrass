using System;
using UnityEngine;
using System.Collections.Generic;
using UnityEditor.U2D.Common;

namespace UnityEditor.U2D.Animation
{
    internal class SpriteSelectorWidget
    {
        class Styles
        {
            public GUIStyle gridListStyle;

            public Styles()
            {
                gridListStyle = new GUIStyle("GridList")
                {
                    alignment = GUI.skin.button.alignment,
                    padding = new RectOffset(k_GridCellPadding, k_GridCellPadding, k_GridCellPadding, k_GridCellPadding),
                    fixedHeight = k_TargetPreviewSize,
                    fixedWidth = k_TargetPreviewSize
                };
            }
        }

        static Texture2D spriteLoadingThumbnail
        {
            get
            {
                if (s_LoadingSpriteTexture == null)
                    s_LoadingSpriteTexture = AssetPreview.GetMiniTypeThumbnail(typeof(Sprite));
                return s_LoadingSpriteTexture;
            }
        }

        static Texture2D s_LoadingSpriteTexture;

        const int k_TargetPreviewSize = 64;
        const int k_GridCellPadding = 2;
        const int k_ScrollbarWidthOffset = 10;

        List<int> m_SpritePreviewNeedFetching = new();

        Sprite[] m_SpriteList = Array.Empty<Sprite>();
        Texture2D[] m_SpritePreviews = Array.Empty<Texture2D>();

        int m_ClientId = 0;
        int m_PreviewCacheSize = 0;

        Vector2 m_ScrollPos;
        Styles m_Style;


        public void Initialize(int clientId)
        {
            m_ClientId = clientId;
        }

        public void Dispose()
        {
            if (m_ClientId != 0)
            {
                InternalEditorBridge.ClearAssetPreviews(m_ClientId);
                m_ClientId = 0;
            }
        }

        public void UpdateContents(Sprite[] sprites)
        {
            m_SpritePreviewNeedFetching.Clear();

            InternalEditorBridge.ClearAssetPreviews(m_ClientId);

            var spriteCount = sprites.Length;
            m_PreviewCacheSize = spriteCount + 1;
            InternalEditorBridge.SetAssetPreviewTextureCacheSize(m_PreviewCacheSize, m_ClientId);

            m_SpriteList = sprites;
            m_SpritePreviews = new Texture2D[spriteCount];

            m_SpritePreviewNeedFetching.Capacity = spriteCount;
            for (var i = 0; i < spriteCount; ++i)
                m_SpritePreviewNeedFetching.Add(i);
        }

        public int ShowGUI(int selectedIndex)
        {
            if (m_Style == null)
                m_Style = new Styles();

            var drawRect = EditorGUILayout.GetControlRect(false, k_TargetPreviewSize + 10f, new[] { GUILayout.ExpandWidth(true) });
            if (Event.current.type == EventType.Repaint)
                GUI.skin.box.Draw(drawRect, false, false, false, false);
            if (m_SpriteList == null || m_SpriteList.Length == 0)
            {
                return selectedIndex;
            }

            selectedIndex = (selectedIndex > m_SpriteList.Length) ? 0 : selectedIndex;

            var widthMargin = (m_Style.gridListStyle.margin.left + m_Style.gridListStyle.margin.right) * 0.5f;
            var heightMargin = (m_Style.gridListStyle.margin.top + m_Style.gridListStyle.margin.bottom) * 0.5f;
            GetRowColumnCount(drawRect.width - k_ScrollbarWidthOffset, k_TargetPreviewSize + Mathf.RoundToInt(widthMargin), m_SpriteList.Length, out var columnCount, out var rowCount);
            if (columnCount > 0 && rowCount > 0)
            {
                var height = rowCount * (k_TargetPreviewSize + heightMargin);
                var width = columnCount * (k_TargetPreviewSize + widthMargin);
                var scrollViewRect = new Rect(drawRect.x - k_ScrollbarWidthOffset, drawRect.y, width, height);
                m_ScrollPos = GUI.BeginScrollView(drawRect, m_ScrollPos, scrollViewRect, false, false, GUIStyle.none, GUI.skin.verticalScrollbar);

                var gridRect = scrollViewRect;
                gridRect.x += (drawRect.width - width - k_ScrollbarWidthOffset) * 0.5f;
                selectedIndex = GUI.SelectionGrid(gridRect, selectedIndex, m_SpritePreviews, columnCount, m_Style.gridListStyle);

                GUI.EndScrollView();
            }

            return selectedIndex;
        }

        static void GetRowColumnCount(float drawWidth, int size, int contentCount, out int columnCount, out int rowCount)
        {
            columnCount = (int)drawWidth / size;
            if (columnCount == 0)
                rowCount = 0;
            else
                rowCount = Math.Max(1, Mathf.CeilToInt((float)contentCount / columnCount));
        }

        public bool UpdateSpritePreviews()
        {
            var remainingPreviewCount = m_SpritePreviewNeedFetching.Count;
            if (remainingPreviewCount == 0)
                return false;

            for (var i = remainingPreviewCount - 1; i >= 0; --i)
            {
                var index = m_SpritePreviewNeedFetching[i];
                if (m_SpriteList[index] == null)
                {
                    m_SpritePreviews[index] = EditorGUIUtility.Load("icons/console.warnicon.png") as Texture2D;
                    m_SpritePreviewNeedFetching.RemoveAt(i);
                }
                else
                {
                    var spriteId = m_SpriteList[index].GetInstanceID();
                    var spritePreview = InternalEditorBridge.GetAssetPreview(spriteId, m_ClientId);
                    if (spritePreview != null)
                    {
                        m_SpritePreviews[index] = spritePreview;
                        m_SpritePreviewNeedFetching.RemoveAt(i);
                    }
                    else
                    {
                        m_SpritePreviews[index] = spriteLoadingThumbnail;
                        if (!InternalEditorBridge.IsLoadingAssetPreview(spriteId, m_ClientId))
                            m_SpritePreviewNeedFetching.RemoveAt(i);
                    }
                }
            }

            return remainingPreviewCount != m_SpritePreviewNeedFetching.Count;
        }
    }
}
