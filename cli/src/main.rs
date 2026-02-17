fn usage_hint() -> &'static str {
    "aidb cli placeholder"
}

fn main() {
    println!("{}", usage_hint());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usage_hint_is_not_empty() {
        assert!(!usage_hint().is_empty());
    }
}
