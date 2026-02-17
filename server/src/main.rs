fn banner() -> &'static str {
    "aidb server placeholder"
}

fn main() {
    println!("{}", banner());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn banner_is_not_empty() {
        assert!(!banner().is_empty());
    }
}
