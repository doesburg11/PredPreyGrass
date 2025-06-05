using System;
using System.Collections.Generic;

using UnityEngine;
using UnityEditor;

using Codice.CM.Common;
using Codice.Client.Common;
using Codice.Client.Common.Authentication;
using Codice.Client.Common.Connection;
using Codice.Client.Common.WebApi.Responses;
using PlasticGui;
using PlasticGui.Configuration.CloudEdition;
using PlasticGui.Configuration.CloudEdition.Welcome;
using PlasticGui.WorkspaceWindow.Home;
using Unity.PlasticSCM.Editor.UI;
using Unity.PlasticSCM.Editor.UI.Progress;

namespace Unity.PlasticSCM.Editor.Configuration
{
    internal class SSOCredentialsDialog :
        PlasticDialog,
        OAuthSignIn.INotify,
        GetCloudOrganizations.INotify
    {
        protected override Rect DefaultRect
        {
            get
            {
                var baseRect = base.DefaultRect;
                return new Rect(baseRect.x, baseRect.y, 525, 450);
            }
        }

        internal static AskCredentialsToUser.DialogData RequestCredentials(
            string cloudServer,
            EditorWindow parentWindow)
        {
            SSOCredentialsDialog dialog = Create(
                cloudServer, new ProgressControlsForDialogs());
            ResponseType dialogResult = dialog.RunModal(parentWindow);

            return dialog.BuildCredentialsDialogData(dialogResult);
        }

        protected override string GetTitle()
        {
            return PlasticLocalization.Name.CredentialsDialogTitle.GetString();
        }

        protected override void OnModalGUI()
        {
            Title(PlasticLocalization.Name.CredentialsDialogTitle.GetString());

            Paragraph(
                PlasticLocalization.Name.CredentialsDialogExplanation.GetString(
                    mOrganizationInfo.DisplayName));

            GUILayout.Space(20);

            DoEntriesArea();

            GUILayout.Space(10);

            DrawProgressForDialogs.For(
                mProgressControls.ProgressData);

            GUILayout.Space(10);

            DoButtonsArea();
        }

        void DoEntriesArea()
        {
            Paragraph("Sign in with Unity ID");
            GUILayout.Space(5);

            DoUnityIDButton();

            GUILayout.Space(25);
            Paragraph("    --or--    ");

            Paragraph("Sign in with email");

            mEmail = TextEntry(
                PlasticLocalization.Name.Email.GetString(),
                mEmail, ENTRY_WIDTH, ENTRY_X);

            GUILayout.Space(5);

            mPassword = PasswordEntry(
                PlasticLocalization.Name.Password.GetString(),
                mPassword, ENTRY_WIDTH, ENTRY_X);
        }

        void DoUnityIDButton()
        {
            if (NormalButton("Sign in with Unity ID"))
            {
                Guid state = Guid.NewGuid();

                OAuthSignInForUnityPackage(
                    GetAuthProviders.GetUnityIdAuthProvider(string.Empty, state),
                    GetCredentialsFromState.Build(
                        string.Empty,
                        state,
                        SEIDWorkingMode.SSOWorkingMode,
                        PlasticGui.Plastic.WebRestAPI));
            }
        }

        void DoButtonsArea()
        {
            using (new EditorGUILayout.HorizontalScope())
            {
                GUILayout.FlexibleSpace();

                if (Application.platform == RuntimePlatform.WindowsEditor)
                {
                    DoOkButton();
                    DoCancelButton();
                    return;
                }

                DoCancelButton();
                DoOkButton();
            }
        }

        void DoOkButton()
        {
            if (!AcceptButton(PlasticLocalization.Name.OkButton.GetString()))
                return;

            OkButtonWithValidationAction();
        }

        void DoCancelButton()
        {
            if (!NormalButton(PlasticLocalization.Name.CancelButton.GetString()))
                return;

            CancelButtonAction();
        }

        void OkButtonWithValidationAction()
        {
            mCredentials = new Credentials(
                new SEID(mEmail, false, mPassword),
                SEIDWorkingMode.LDAPWorkingMode);

            GetCloudOrganizations.GetOrganizationsInThreadWaiter(
                mCredentials.User.Data,
                mCredentials.User.Password,
                mProgressControls,
                this,
                PlasticGui.Plastic.WebRestAPI,
                CmConnection.Get());
        }

        void OAuthSignInForUnityPackage(
            AuthProvider authProvider, IGetCredentialsFromState getCredentialsFromState)
        {
            OAuthSignIn oAuthSignIn = new OAuthSignIn();

            oAuthSignIn.SignInForProviderInThreadWaiter(
                authProvider,
                string.Empty,
                mProgressControls,
                this,
                new OAuthSignIn.Browser(),
                getCredentialsFromState);
        }

        void OAuthSignIn.INotify.SignedInForCloud(
            string chosenProviderName, Credentials credentials)
        {
            mCredentials = credentials;

            GetCloudOrganizations.GetOrganizationsInThreadWaiter(
                mCredentials.User.Data,
                mCredentials.User.Password,
                mProgressControls,
                this,
                PlasticGui.Plastic.WebRestAPI,
                CmConnection.Get());
        }

        void OAuthSignIn.INotify.SignedInForOnPremise(
            string server, string proxy, Credentials credentials)
        {
            // The Plugin does not support SSO for on-premise (OIDCWorkingMode / SAMLWorkingMode)
            // as it is not prepared to show the necessary UI
        }

        void OAuthSignIn.INotify.Cancel(string errorMessage)
        {
            CancelButtonAction();
        }

        void GetCloudOrganizations.INotify.CloudOrganizationsRetrieved(
            List<string> cloudOrganizations)
        {
            if (!cloudOrganizations.Contains(mOrganizationInfo.Server))
            {
                CancelButtonAction();
                return;
            }

            ClientConfiguration.Save(
                mOrganizationInfo.Server,
                mCredentials.Mode,
                mCredentials.User.Data,
                mCredentials.User.Password);

            GetWindow<PlasticWindow>().InitializePlastic();
            OkButtonAction();
        }

        void GetCloudOrganizations.INotify.Error(ErrorResponse.ErrorFields error)
        {
            CancelButtonAction();
        }

        AskCredentialsToUser.DialogData BuildCredentialsDialogData(ResponseType dialogResult)
        {
            return dialogResult == ResponseType.Ok
                ? AskCredentialsToUser.DialogData.Success(mCredentials)
                : AskCredentialsToUser.DialogData.Failure(SEIDWorkingMode.SSOWorkingMode);
        }

        static SSOCredentialsDialog Create(
            string server,
            ProgressControlsForDialogs progressControls)
        {
            var instance = CreateInstance<SSOCredentialsDialog>();
            instance.mOrganizationInfo = OrganizationsInformation.FromServer(server);
            instance.mProgressControls = progressControls;
            instance.mEnterKeyAction = instance.OkButtonWithValidationAction;
            instance.mEscapeKeyAction = instance.CancelButtonAction;
            return instance;
        }

        string mEmail;
        string mPassword = string.Empty;

        Credentials mCredentials;
        ProgressControlsForDialogs mProgressControls;

        OrganizationInfo mOrganizationInfo;

        const float ENTRY_WIDTH = 345f;
        const float ENTRY_X = 150f;
    }
}
