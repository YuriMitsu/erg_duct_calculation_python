#!/bin/bash
cat << EOF|/Applications/harris/idl86/bin/idl
!path=expand_path('+/Users/ampuku/Documents/duct/code/IDL')+':'+expand_path('+/Users/ampuku/IDLplagin/spedas_4_1')+':'+!path
.r /Users/ampuku/Documents/duct/code/IDL/tplots/kpara_LASVD_ma3_mask_tplots/test.pro
test
EOF