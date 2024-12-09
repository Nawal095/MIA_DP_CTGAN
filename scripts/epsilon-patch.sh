#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <epsilon_value>"
    exit 1
fi

# Assign arguments to variables
epsilon_value=$1

# Get the path to the dp-cgans package
package_path=$(pip show dp-cgans | grep Location | awk '{print $2}')/dp_cgans

# Find the file containing the specified code snippet
file_path=$(grep -rl "if self.private:" $package_path)

echo "File Path: $file_path"

if [ -z "$file_path" ]; then
    echo "File containing the specified code snippet not found."
    exit 1
fi

# Check if the file contains "delta = 2e-6" with any amount of whitespace
if grep -q "epsilon, _, opt_order = get_privacy_spent(orders, rdp, target_delta=delta)" "$file_path"; then
    # Create a temporary patch file with the new epsilon value
    patch_file=$(mktemp)
    cat <<EOF > $patch_file
diff --git a/$file_path b/$file_path
index cbc7f60..48a5aa7 100644
--- a/$file_path
+++ b/$file_path
@@ -589,12 +589,12 @@ class DPCGANSynthesizer(BaseSynthesizer):
                         if self.private:
                             orders = [1 + x / 10. for x in range(1, 100)]
                             sampling_probability = self._batch_size/len(train_data)
-                            delta = 2e-6
+                            epsilon = $epsilon_value
                             rdp = compute_rdp(q=sampling_probability,
                                                 noise_multiplier=sigma,
                                                 steps=i * steps_per_epoch,
                                                 orders=orders)
-                            epsilon, _, opt_order = get_privacy_spent(orders, rdp, target_delta=delta) # target_delta=1e-5
+                            _, delta, opt_order = get_privacy_spent(orders, rdp, target_eps=epsilon)
 
                             print('differential privacy with eps = {:.3g} and delta = {}.'.format(
                                 epsilon, delta))
EOF

    # Apply the patch without prompting for user input
    patch -p1 < $patch_file

    # Clean up the temporary patch file
    rm $patch_file
    rm "$file_path.orig"

    echo "Patch applied successfully with epsilon = $epsilon_value to $file_path"
else
    # Update the epsilon value in the file, handling different formats
    perl -pi -e "s/epsilon\s*=\s*[0-9.eE+-]+/epsilon = $epsilon_value/" $file_path
    echo "Epsilon value updated to $epsilon_value in $file_path"
fi
