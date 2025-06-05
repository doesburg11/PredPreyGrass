using System;
using UnityEngine;

namespace UnityEditor.Tilemaps
{
    internal class GridPaletteWhiteboxPalettesDropdown : FlexibleMenu
    {
        public static bool IsOpen = false;

        public GridPaletteWhiteboxPalettesDropdown(IFlexibleMenuItemProvider itemProvider, int selectionIndex, FlexibleMenuModifyItemUI modifyItemUi, Action<int, object> itemClickedCallback, float minWidth)
            : base(itemProvider, selectionIndex, modifyItemUi, itemClickedCallback)
        {
            minTextWidth = minWidth;
        }

        public override void OnOpen()
        {
            base.OnOpen();
            IsOpen = true;
        }

        public override void OnClose()
        {
            IsOpen = false;
            base.OnClose();
        }

        internal class MenuItemProvider : IFlexibleMenuItemProvider
        {
            public int Count()
            {
                return TilePaletteWhiteboxSamplesUtility.whiteboxSampleNames.Count;
            }

            public object GetItem(int index)
            {
                if (index < TilePaletteWhiteboxSamplesUtility.whiteboxSampleNames.Count)
                    return TilePaletteWhiteboxSamplesUtility.whiteboxSampleNames[index];
                return null;
            }

            public int Add(object obj)
            {
                throw new NotImplementedException();
            }

            public void Replace(int index, object newPresetObject)
            {
                throw new NotImplementedException();
            }

            public void Remove(int index)
            {
                throw new NotImplementedException();
            }

            public object Create()
            {
                throw new NotImplementedException();
            }

            public void Move(int index, int destIndex, bool insertAfterDestIndex)
            {
                throw new NotImplementedException();
            }

            public string GetName(int index)
            {
                if (index < TilePaletteWhiteboxSamplesUtility.whiteboxSampleNames.Count)
                    return TilePaletteWhiteboxSamplesUtility.whiteboxSampleNames[index];
                return "";
            }

            public bool IsModificationAllowed(int index)
            {
                return false;
            }

            public int[] GetSeperatorIndices()
            {
                return Array.Empty<int>();
            }
        }
    }
}
