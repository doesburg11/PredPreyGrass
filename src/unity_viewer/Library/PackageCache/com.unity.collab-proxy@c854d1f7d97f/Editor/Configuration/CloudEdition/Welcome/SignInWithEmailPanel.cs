using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UIElements;

using Codice.Client.Common;
using Codice.Client.Common.Authentication;
using Codice.Client.Common.WebApi;
using Codice.Client.Common.WebApi.Responses;
using Codice.CM.Common;
using PlasticGui;
using PlasticGui.Configuration.CloudEdition.Welcome;
using PlasticGui.Configuration.CloudEdition;
using Unity.PlasticSCM.Editor.UI.UIElements;

namespace Unity.PlasticSCM.Editor.Configuration.CloudEdition.Welcome
{
    internal class SignInWithEmailPanel : VisualElement, GetCloudOrganizations.INotify
    {
        internal SignInWithEmailPanel(
            IWelcomeWindowNotify notify,
            IPlasticWebRestApi restApi)
        {
            mNotify = notify;
            mRestApi = restApi;

            InitializeLayoutAndStyles();

            BuildComponents();
        }

        internal void Dispose()
        {
            mSignInButton.clicked -= SignInButton_Clicked;
            mBackButton.clicked -= BackButton_Clicked;
            mSignUpButton.clicked -= SignUpButton_Clicked;
        }

        void ShowSignUpNeeded()
        {
            mSignUpNeededNotificationContainer.Show();
        }

        void HideSignUpNeeded()
        {
            mSignUpNeededNotificationContainer.Collapse();
        }

        void GetCloudOrganizations.INotify.CloudOrganizationsRetrieved(
            List<string> cloudServers)
        {
            mNotify.ProcessLoginResponseWithOrganizations(mCredentials, cloudServers);
        }

        void GetCloudOrganizations.INotify.Error(
            ErrorResponse.ErrorFields error)
        {
            if (error.ErrorCode == ErrorCodes.UserNotFound)
            {
                ShowSignUpNeeded();
                return;
            }

            HideSignUpNeeded();
            ((IProgressControls)mProgressControls).ShowError(error.Message);
        }

        void CleanNotificationLabels()
        {
            mEmailNotificationLabel.text = string.Empty;
            mPasswordNotificationLabel.text = string.Empty;

            HideSignUpNeeded();
        }

        void SignInButton_Clicked()
        {
            CleanNotificationLabels();

            ValidateEmailAndPassword.ValidationResult validationResult;
            if (!ValidateEmailAndPassword.IsValid(mEmailField.text, mPasswordField.text, out validationResult))
            {
                ShowValidationResult(validationResult);
                return;
            }

            mCredentials = new Credentials(
                new SEID(mEmailField.text, false, mPasswordField.text),
                SEIDWorkingMode.LDAPWorkingMode);

            GetCloudOrganizations.GetOrganizationsInThreadWaiter(
                mCredentials.User.Data,
                mCredentials.User.Password,
                mProgressControls,
                this,
                mRestApi,
                CmConnection.Get());
        }

        void ShowValidationResult(
            ValidateEmailAndPassword.ValidationResult validationResult)
        {
            if (validationResult.UserError != null)
            {
                mEmailNotificationLabel.text = validationResult.UserError;
            }

            if (validationResult.PasswordError != null)
            {
                mPasswordNotificationLabel.text = validationResult.PasswordError;
            }
        }

        void BackButton_Clicked()
        {
            mNotify.Back();
        }

        void InitializeLayoutAndStyles()
        {
            this.LoadLayout(typeof(SignInWithEmailPanel).Name);
            this.LoadStyle(typeof(SignInWithEmailPanel).Name);
        }

        void SignUpButton_Clicked()
        {
            Application.OpenURL(UnityUrl.DevOps.GetSignUp());
        }

        void BuildComponents()
        {
            mEmailField = this.Q<TextField>("email");
            mPasswordField = this.Q<TextField>("password");
            mEmailNotificationLabel = this.Q<Label>("emailNotification");
            mPasswordNotificationLabel = this.Q<Label>("passwordNotification");
            mSignInButton = this.Q<Button>("signIn");
            mBackButton = this.Q<Button>("back");
            mSignUpButton = this.Q<Button>("signUpButton");
            mProgressContainer = this.Q<VisualElement>("progressContainer");
            mSignUpNeededNotificationContainer = this.Q<VisualElement>("signUpNeededNotificationContainer");

            mSignInButton.clicked += SignInButton_Clicked;
            mBackButton.clicked += BackButton_Clicked;
            mSignUpButton.clicked += SignUpButton_Clicked;
            mEmailField.FocusOnceLoaded();

            mProgressControls = new ProgressControlsForDialogs(new VisualElement[] { mSignInButton });
            mProgressContainer.Add((VisualElement)mProgressControls);

            this.SetControlText<Label>("signInLabel",
                PlasticLocalization.Name.SignInWithEmail);
            this.SetControlLabel<TextField>("email",
                PlasticLocalization.Name.Email);
            this.SetControlLabel<TextField>("password",
                PlasticLocalization.Name.Password);
            this.SetControlText<Button>("signIn",
                PlasticLocalization.Name.SignIn);
            this.SetControlText<Button>("back",
                PlasticLocalization.Name.BackButton);
            this.SetControlText<Label>("signUpNeededNotificationLabel",
                PlasticLocalization.Name.SignUpNeededNoArgs);
            this.SetControlText<Button>("signUpButton",
                PlasticLocalization.Name.SignUp);
        }

        TextField mEmailField;
        TextField mPasswordField;

        Label mEmailNotificationLabel;
        Label mPasswordNotificationLabel;

        Button mSignInButton;
        Button mBackButton;
        Button mSignUpButton;

        VisualElement mProgressContainer;
        VisualElement mSignUpNeededNotificationContainer;

        Credentials mCredentials;
        ProgressControlsForDialogs mProgressControls;

        readonly IWelcomeWindowNotify mNotify;
        readonly IPlasticWebRestApi mRestApi;
    }
}
