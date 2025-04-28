using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PoopDetector.Models
{
    public class PermissionRequestManager
    {
        public static async Task<bool> RequestPermissions(bool withMic = false, bool withStorageWrite = false)
        {

            var status = await Permissions.CheckStatusAsync<Permissions.LocationWhenInUse>();
            if (status != PermissionStatus.Granted)
            {
                status = await Permissions.RequestAsync<Permissions.LocationWhenInUse>();
                if (status != PermissionStatus.Granted) return false;
            }


            return true;
        }
    }
}
