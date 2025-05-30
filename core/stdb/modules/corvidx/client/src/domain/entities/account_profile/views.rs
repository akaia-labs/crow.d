use std::fmt::{self, Display, Formatter};

use crate::common::{
	presentation::Displayable,
	stdb::{AccountProfile, AccountProfileName},
};

impl Display for AccountProfileName {
	fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
		if let Some(name_extension) = &self.name_extension {
			write!(formatter, "{} {}", self.short_name, name_extension)
		} else {
			write!(formatter, "{}", self.short_name)
		}
	}
}

impl Displayable for AccountProfile {
	fn display_name(&self) -> String {
		self.metadata.name.to_string()
	}
}
