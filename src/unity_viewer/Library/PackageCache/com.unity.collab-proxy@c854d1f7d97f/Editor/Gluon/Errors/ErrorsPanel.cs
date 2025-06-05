using System.Collections.Generic;

using UnityEditor;
using UnityEngine;

using Codice.CM.Common;
using PlasticGui;
using Unity.PlasticSCM.Editor.UI;
using Unity.PlasticSCM.Editor.UI.Tree;

namespace Unity.PlasticSCM.Editor.Gluon.Errors
{
    internal class ErrorsPanel
    {
        internal bool IsVisible { get; private set; }

        internal ErrorsPanel(
            string title,
            string treeSettingsName)
        {
            mTitle = title;

            ErrorsListHeaderState errorsListHeaderState =
                ErrorsListHeaderState.GetDefault();
            TreeHeaderSettings.Load(
                errorsListHeaderState,
                treeSettingsName,
                UnityConstants.UNSORT_COLUMN_ID);

            mErrorsListView = new ErrorsListView(errorsListHeaderState);
            mErrorsListView.Reload();

            mErrorDetailsSplitterState = PlasticSplitterGUILayout.InitSplitterState(
                new float[] { 0.60f, 0.40f },
                new int[] { 100, 100 },
                new int[] { 100000, 100000 }
            );
        }

        internal void UpdateErrorsList(List<ErrorMessage> errorMessages)
        {
            mErrorsListView.BuildModel(errorMessages);
            mErrorsListView.Reload();

            IsVisible = errorMessages.Count > 0;
        }

        internal void OnDisable()
        {
            TreeHeaderSettings.Save(
                mErrorsListView.multiColumnHeader.state,
                UnityConstants.GLUON_INCOMING_ERRORS_TABLE_SETTINGS_NAME);
        }

        internal void OnGUI()
        {
            EditorGUILayout.BeginVertical();

            DrawSplitter.ForHorizontalIndicator();
            DoErrorsListArea(mErrorsListView, mErrorDetailsSplitterState);

            EditorGUILayout.EndVertical();
        }

        void DoErrorsListArea(
            ErrorsListView errorsListView,
            object splitterState)
        {
            EditorGUILayout.BeginVertical();

            GUILayout.Label(
                mTitle,
                EditorStyles.boldLabel);

            DoErrorsListSplitViewArea(
                errorsListView, splitterState);

            EditorGUILayout.EndVertical();
        }

        void DoErrorsListSplitViewArea(
            ErrorsListView errorsListView,
            object splitterState)
        {
            EditorGUILayout.BeginHorizontal();

            PlasticSplitterGUILayout.BeginHorizontalSplit(splitterState);

            DoErrorsListViewArea(errorsListView);

            DoErrorDetailsTextArea(errorsListView.GetSelectedError());

            PlasticSplitterGUILayout.EndHorizontalSplit();

            EditorGUILayout.EndHorizontal();
        }

        static void DoErrorsListViewArea(
            ErrorsListView errorsListView)
        {
            Rect treeRect = GUILayoutUtility.GetRect(0, 100000, 0, 100000);

            errorsListView.OnGUI(treeRect);
        }

        void DoErrorDetailsTextArea(ErrorMessage selectedErrorMessage)
        {
            string errorDetailsText = selectedErrorMessage == null ?
                string.Empty : selectedErrorMessage.Error;

            mErrorDetailsScrollPosition = GUILayout.BeginScrollView(
                mErrorDetailsScrollPosition);

            GUILayout.TextArea(
                errorDetailsText, UnityStyles.TextFieldWithWrapping,
                GUILayout.ExpandHeight(true));

            GUILayout.EndScrollView();
        }

        Vector2 mErrorDetailsScrollPosition;

        readonly string mTitle;
        readonly ErrorsListView mErrorsListView;
        readonly object mErrorDetailsSplitterState;
    }
}
