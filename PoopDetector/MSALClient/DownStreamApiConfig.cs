// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace SignInMaui.MSALClient
{
    public class DownStreamApiConfig
    {
        /// <summary>
        /// Gets or sets the scopes for MS graph call.
        /// </summary>
        /// <value>
        /// The scopes.
        /// </value>
        public string Scopes { get; set; }
        public string Domain { get; set; }
        public string AppId { get; set; }
        /// <summary>
        /// Gets the scopes in a format as expected by the various MSAL SDK methods.
        /// </summary>
        /// <value>
        /// The scopes.
        /// </value>
        public string[] ScopesArray
        {
            get
            {if (Scopes == string.Empty) return [];
                return Scopes.Split(' ');
            }
        }
        public string[] FullNameScopesArray
        {
            get
            {
                return ScopesArray.Select(x => $"https://{Domain}/{AppId}/{x}").ToArray();
            }
        }
    }
}
