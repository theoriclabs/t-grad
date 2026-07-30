import Tgrad.Runtime.Hip

-- This file must not compile: the vendor probe credential constructor is
-- private, so caller-provided profile facts cannot mint shared availability.
#check Tgrad.Runtime.Hip.ProbeCredential.mk
